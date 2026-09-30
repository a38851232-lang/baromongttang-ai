from __future__ import annotations

import base64
import ipaddress
import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import requests
from flask import Flask, jsonify, request

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("baromongttang-ai")

VERSION = "4.0.0"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_IMAGE_DETAIL = os.getenv("OPENAI_IMAGE_DETAIL", "auto")
CALLBACK_MODE = os.getenv("CALLBACK_MODE", "auto").lower()
SYNC_BUDGET = float(os.getenv("SYNC_BUDGET", "3.5"))
KEEP_ALIVE_INTERVAL = float(os.getenv("KEEP_ALIVE_INTERVAL", "240.0"))

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "none").strip().lower()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "chat-images").strip()

_has_pil = False
try:
    from PIL import Image
    _has_pil = True
except Exception:
    pass

app = Flask(__name__)

def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")

def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val.strip())
    except Exception:
        return default

def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    return {}

def _as_list(obj: Any) -> List[Any]:
    if isinstance(obj, list):
        return obj
    return []

def _model_candidates() -> List[str]:
    m = OPENAI_MODEL
    candidates = [m]
    if "gpt-4o" in m:
        candidates.extend(["gpt-4o", "gpt-4o-mini", "chatgpt-4o-latest"])
    elif "gpt-4" in m:
        candidates.extend(["gpt-4o", "gpt-4-turbo"])
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique

def _image_value(value, depth=0):
    if depth > 6:
        return None
    if isinstance(value, dict):
        for key in ('secureUrls', 'origin', 'value', 'url', 'imageUrl'):
            found = _image_value(value.get(key), depth + 1)
            if found:
                return found
        return None
    if isinstance(value, list):
        return next((u for v in value if (u := _image_value(v, depth + 1))), None)
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value.startswith(('{', '[', '"')):
        try:
            return _image_value(json.loads(value), depth + 1)
        except (ValueError, TypeError):
            return None
    if value.startswith('List(') and value.endswith(')'):
        value = re.split(r',\s*(?=https?://)', value[5:-1])[0].strip()
    try:
        parts = urlsplit(value)
        if parts.scheme in ('http', 'https') and parts.hostname and not re.search(r'\s', value):
            return value
    except ValueError:
        pass
    return None

def _openai_image_input(image_url: str) -> str:
    """Make Kakao's short-lived CDN image available to the vision model.

    Public image URLs can go directly to OpenAI.  Kakao secure images are
    fetched while their signed URL is valid and sent as image bytes instead.
    """
    hostname = (urlsplit(image_url).hostname or '').lower()
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise ValueError('non-public image host')
    kakao_hosts = ('kakao.com', 'kakaocdn.net', 'daumcdn.net')
    if not any(hostname == h or hostname.endswith('.' + h) for h in kakao_hosts):
        return image_url

    try:
        response = requests.get(image_url, timeout=(3, 10), stream=True,
                                allow_redirects=False)
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_content(64 * 1024):
            size += len(chunk)
            if size > 10 * 1024 * 1024:
                raise ValueError('image too large')
            chunks.append(chunk)
        if not size:
            raise ValueError('empty image')
        data = b''.join(chunks)
        if data.startswith(b'\xff\xd8\xff'):
            mime = 'image/jpeg'
        elif data.startswith(b'\x89PNG\r\n\x1a\n'):
            mime = 'image/png'
        elif data.startswith((b'GIF87a', b'GIF89a')):
            mime = 'image/gif'
        elif data.startswith(b'RIFF') and data[8:12] == b'WEBP':
            mime = 'image/webp'
        else:
            raise ValueError('unsupported image bytes')
        return f'data:{mime};base64,{base64.b64encode(data).decode("ascii")}'
    finally:
        if 'response' in locals():
            response.close()

def extract_image_url(payload):
    payload = payload if isinstance(payload, dict) else {}
    action = payload.get('action')
    action = action if isinstance(action, dict) else {}
    # Only configured image keys; do not select an unrelated website parameter.
    keys = ('secureimage', 'image', 'image_url', 'photo')
    for section in ('params', 'detailParams'):
        params = action.get(section)
        if isinstance(params, dict):
            for key in keys:
                found = _image_value(params.get(key))
                if found:
                    return found, f'action.{section}.{key}'
    contexts = payload.get('contexts')
    for context in contexts if isinstance(contexts, list) else []:
        params = context.get('params') if isinstance(context, dict) else None
        if isinstance(params, dict):
            for key in keys:
                found = _image_value(params.get(key))
                if found:
                    return found, f'contexts.params.{key}'
    user = payload.get('userRequest')
    utterance = user.get('utterance') if isinstance(user, dict) else None
    if isinstance(utterance, str):
        for candidate in re.findall(r'https?://[^\s<>\"]+', utterance):
            found = _image_value(candidate)
            if found:
                return found, 'utterance_url'
    return None, 'not_found'

def analyze_image_with_openai(image_url: str, user_text: str) -> str:
    if not OPENAI_API_KEY:
        return "âš ï¸ OpenAI API í‚¤ê°€ ì„¤ì •ë˜ì–´ ìžˆì§€ ì•ŠìŠµë‹ˆë‹¤. Render í™˜ê²½ë³€ìˆ˜ë¥¼ í™•ì¸í•´ì£¼ì„¸ìš”."

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    
    system_prompt = (
        "ë‹¹ì‹ ì€ ì‚¬ì§„ì„ ì‹¤ì œë¡œ ì‚´íŽ´ë³´ê³  í•œêµ­ì–´ë¡œ ì‰½ê³  ê°„ê²°í•˜ê²Œ ë‹µí•˜ëŠ” AI ì§€ìŒìž…ë‹ˆë‹¤. "
        "ì‚¬ì§„ì— ë¬´ì—‡ì´ ìžˆëŠ”ì§€ ë¨¼ì € ì‹ë³„í•˜ê³  ë³´ì´ëŠ” ê¸€ìžë¥¼ ì½ìœ¼ì„¸ìš”. "
        "ì œí’ˆì´ë©´ ì œí’ˆëª…Â·ì¢…ë¥˜Â·ìš©ë„ì™€ ì‚¬ì§„ì—ì„œ í™•ì¸ë˜ëŠ” ì •ë³´ë¥¼ ì„¤ëª…í•˜ì„¸ìš”. "
        "ê½ƒ, ìŒì‹, ìƒí™œìš©í’ˆ, í˜„íŒ ë“± ë‹¤ë¥¸ ì‚¬ì§„ë„ ì‚¬ì§„ì˜ ë‚´ìš©ì— ë§žê²Œ ì„¤ëª…í•˜ì„¸ìš”. "
        "í™•ì¸í•  ìˆ˜ ì—†ëŠ” ì„±ë¶„Â·ê°€ê²©Â·íš¨ëŠ¥Â·ì¶œì „ ë“±ì€ ì¶”ì¸¡í•˜ì§€ ë§ê³  ë¶ˆí™•ì‹¤í•˜ë‹¤ê³  ë°ížˆì„¸ìš”."
    )
    
    try:
        image_input = _openai_image_input(image_url)
    except (requests.RequestException, ValueError) as exc:
        logger.error("image_fetch_failed exception=%s", type(exc).__name__)
        return "âš ï¸ ì‚¬ì§„ ì£¼ì†Œë¥¼ ì½ì§€ ëª»í–ˆìŠµë‹ˆë‹¤. ì‚¬ì§„ì„ ë‹¤ì‹œ ë³´ë‚´ ì£¼ì„¸ìš”."

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text or "ì´ ì‚¬ì§„ì— ë¬´ì—‡ì´ ë³´ì´ëŠ”ì§€ ì„¤ëª…í•´ ì£¼ì„¸ìš”."},
                    {"type": "image_url", "image_url": {"url": image_input, "detail": OPENAI_IMAGE_DETAIL}}
                ]
            }
        ],
        "max_tokens": 1000
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=25)
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"].get("content")
            if not isinstance(content, str) or not content.strip():
                logger.error("openai_failed empty_content")
                return "âš ï¸ ì´ë¯¸ì§€ ë¶„ì„ ë‹µë³€ì´ ë¹„ì–´ ìžˆìŠµë‹ˆë‹¤. ìž ì‹œ í›„ ë‹¤ì‹œ ì‹œë„í•´ ì£¼ì„¸ìš”."
            return content.strip()
        else:
            logger.error("openai_failed http=%s", response.status_code)
            return f"âš ï¸ OpenAI ë¶„ì„ ì¤‘ ì˜¤ë¥˜ê°€ ë°œìƒí–ˆìŠµë‹ˆë‹¤. (ìƒíƒœì½”ë“œ: {response.status_code})"
    except Exception as e:
        logger.error("openai_failed exception=%s", type(e).__name__)
        return "âš ï¸ ì´ë¯¸ì§€ ë¶„ì„ ì¤‘ ì„œë²„ ì˜¤ë¥˜ê°€ ë°œìƒí–ˆìŠµë‹ˆë‹¤."

def send_callback(callback_url: str, text: str):
    if not callback_url:
        return
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": text[:1000]
                    }
                }
            ]
        }
    }
    try:
        response = requests.post(callback_url, json=payload, timeout=5, allow_redirects=False)
        if not 200 <= response.status_code < 300:
            logger.error("callback_failed http=%s", response.status_code)
            return False
        data = response.json()
        status = data.get("status") if isinstance(data, dict) else None
        if status != "SUCCESS":
            logger.error("callback_failed result=%s", status if status in ("FAIL", "ERROR") else "invalid_response")
            return False
        logger.info("callback_success")
        return True
    except (requests.RequestException, ValueError) as exc:
        logger.error("callback_failed exception=%s", type(exc).__name__)
        return False

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "version": VERSION,
        "openai_key_present": bool(OPENAI_API_KEY)
    })

@app.route("/kakao/photo", methods=["POST"])
def kakao_photo():
    payload = _as_dict(request.get_json(silent=True))
    user_request = _as_dict(payload.get("userRequest"))
    callback_url = user_request.get("callbackUrl")
    callback_url = callback_url if isinstance(callback_url, str) and callback_url.startswith("https://") else None
    utterance = user_request.get("utterance")
    utterance = utterance if isinstance(utterance, str) else ""
    
    image_url, src_type = extract_image_url(payload)
    logger.info("photo_request source=%s image_found=%s callback_present=%s", src_type, bool(image_url), bool(callback_url))
    
    if not image_url:
        return jsonify({
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "ì‚¬ì§„ì„ ì°¾ì§€ ëª»í–ˆìŠµë‹ˆë‹¤. ì´ë¯¸ì§€ë¥¼ í¬í•¨í•˜ì—¬ ë‹¤ì‹œ ì „ì†¡í•´ ì£¼ì„¸ìš”."
                        }
                    }
                ]
            }
        })

    # ë™ê¸° ì²˜ë¦¬ ì‹œê°„ ì´ˆê³¼ ë°©ì§€ë¥¼ ìœ„í•œ ì½œë°± ëª¨ë“œ ë¶„ê¸°
    if callback_url:
        def background_work():
            result_text = analyze_image_with_openai(image_url, utterance)
            send_callback(callback_url, result_text)
            
        t = threading.Thread(target=background_work, daemon=True)
        t.start()
        
        return jsonify({"version": "2.0", "useCallback": True})
    # Keep the existing synchronous path when the block has no callback URL.
    result_text = analyze_image_with_openai(image_url, utterance)
    return jsonify({"version": "2.0", "template": {"outputs": [{"simpleText": {
        "text": result_text[:1000]
    }}]}})

def _start_keepalive():
    def loop():
        import time
        while True:
            time.sleep(KEEP_ALIVE_INTERVAL)
            try:
                requests.get(f"http://127.0.0.1:{os.getenv('PORT', '10000')}/health", timeout=5)
            except Exception:
                pass
    threading.Thread(target=loop, daemon=True).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    if not _env_bool("FLASK_DEBUG", False):
        _start_keepalive()
    app.run(host="0.0.0.0", port=port, debug=_env_bool("FLASK_DEBUG", False), threaded=True)
else:
    _start_keepalive()

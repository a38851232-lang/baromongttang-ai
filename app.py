from __future__ import annotations

import base64
import ipaddress
import json
import logging
import os
import re
import threading
import time
import uuid
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

VERSION = "4.0.1"
PHOTO_SLOTS = threading.BoundedSemaphore(4)

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
        logger.info("image_fetch_skipped reason=non_kakao_host host=%s", hostname)
        return image_url

    logger.info("image_fetch_started host=%s", hostname)
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
        logger.info("image_fetch_success host=%s bytes=%s mime=%s", hostname, size, mime)
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
        return "⚠️ OpenAI API 키가 설정되어 있지 않습니다. Render 환경변수를 확인해 주세요."

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "당신은 사진을 실제로 살펴보고 한국어로 쉽고 간결하게 답하는 AI 지음입니다. "
        "사진에 무엇이 있는지 먼저 식별하고 보이는 글자를 읽으세요. "
        "제품이면 제품명·종류·용도와 사진에서 확인되는 정보를 설명하세요. "
        "꽃, 음식, 생활용품, 현판 등 다른 사진도 사진의 내용에 맞게 설명하세요. "
        "확인할 수 없는 성분·가격·효능·출처 등은 추측하지 말고 불확실하다고 밝히세요."
    )

    try:
        image_input = _openai_image_input(image_url)
    except (requests.RequestException, ValueError) as exc:
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
        logger.error("image_fetch_failed exception=%s http=%s", type(exc).__name__, status)
        return "⚠️ 사진 주소를 읽지 못했습니다. 사진을 다시 보내 주세요."

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_text or "이 사진에 무엇이 보이는지 설명해 주세요."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_input,
                            "detail": OPENAI_IMAGE_DETAIL
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }

    logger.info("ai_request_started model=%s detail=%s", OPENAI_MODEL, OPENAI_IMAGE_DETAIL)
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=25,
        )
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"].get("content")
            if not isinstance(content, str) or not content.strip():
                logger.error("ai_response_failed reason=empty_content")
                return "⚠️ 이미지 분석 답변이 비어 있습니다. 잠시 후 다시 시도해 주세요."
            logger.info("ai_response_success chars=%s", len(content))
            return content.strip()

        logger.error("ai_response_failed http=%s", response.status_code)
        return f"⚠️ OpenAI 분석 중 오류가 발생했습니다. (상태코드: {response.status_code})"
    except Exception as exc:
        logger.error("ai_response_failed exception=%s", type(exc).__name__)
        return "⚠️ 이미지 분석 중 서버 오류가 발생했습니다."


def kakao_text_response(text: str):
    """Return a Kakao i Open Builder 2.0 simpleText response."""
    clean = str(text or "").strip() or "처리 결과가 없습니다."
    return jsonify({
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": clean[:1000]
                    }
                }
            ]
        }
    })


def send_callback(callback_url: str, text: str, callback_token: Optional[str] = None):
    if not callback_url:
        return
    headers = {"Content-Type": "application/json"}
    if callback_token:
        headers["X-Kakao-Callback-Token"] = callback_token
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
        response = requests.post(callback_url, json=payload, headers=headers, timeout=10, allow_redirects=False)
        if not 200 <= response.status_code < 300:
            logger.error("callback_failed http=%s", response.status_code)
            return False
        data = response.json()
        status = data.get("status") if isinstance(data, dict) else None
        if status != "SUCCESS":
            message = data.get("message") if isinstance(data, dict) else None
            logger.error("callback_failed result=%s message=%s", status, str(message)[:120])
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
    request_id = uuid.uuid4().hex[:12]
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        logger.warning("photo_invalid_request request_id=%s", request_id)
        return kakao_text_response("요청 형식이 올바르지 않습니다."), 400
    user_request = _as_dict(payload.get("userRequest"))
    logger.info("request_received content_type=%s body_present=%s", request.content_type, bool(payload))

    callback_url = user_request.get("callbackUrl")
    callback_url = (
        callback_url
        if isinstance(callback_url, str) and callback_url.startswith("https://")
        else None
    )

    utterance = user_request.get("utterance")
    utterance = utterance if isinstance(utterance, str) else ""

    callback_token = request.headers.get("X-Kakao-Callback-Token") or user_request.get("callbackToken")
    callback_token = callback_token if isinstance(callback_token, str) and callback_token else None

    # Only this request's image; never reuse a previous context's photo.
    image_url, src_type = extract_image_url({
        "action": payload.get("action"), "userRequest": user_request
    })
    logger.info(
        "secureimage_received source=%s image_found=%s callback_present=%s callback_token_present=%s",
        src_type,
        bool(image_url),
        bool(callback_url),
        bool(callback_token),
    )

    if not image_url:
        action = _as_dict(payload.get("action"))
        logger.warning(
            "secureimage_missing params_keys=%s detail_params_keys=%s",
            list(_as_dict(action.get("params")).keys()),
            list(_as_dict(action.get("detailParams")).keys()),
        )
        return kakao_text_response(
            "사진 정보를 서버에서 찾지 못했습니다. "
            "카카오의 ‘사진으로 묻기’에서 사진을 다시 보내 주세요."
        )

    if not callback_url:
        # Preserve the synchronous response path supplied by the final diff.
        result_text = analyze_image_with_openai(image_url, utterance)
        logger.info("kakao_response_ready mode=sync chars=%s", len(result_text))
        return kakao_text_response(result_text)

    # Acknowledge before expensive image download / AI processing.
    if callback_url:
        if not PHOTO_SLOTS.acquire(blocking=False):
            return kakao_text_response("사진 처리 요청이 많습니다. 잠시 후 다시 보내 주십시오.")

        def background_work():
            started = time.monotonic()
            try:
                logger.info("photo_analysis_started request_id=%s", request_id)
                try:
                    result_text = analyze_image_with_openai(image_url, utterance)
                except Exception as exc:
                    logger.error("photo_analysis_failed request_id=%s kind=%s", request_id, type(exc).__name__)
                    result_text = "사진 분석 중 오류가 발생했습니다. 잠시 후 다시 보내 주십시오."
                logger.info("photo_analysis_finished request_id=%s elapsed=%.2f error_response=%s",
                            request_id, time.monotonic() - started, result_text.startswith("⚠️"))
                delivered = send_callback(callback_url, result_text, callback_token)
                logger.info("photo_delivery request_id=%s success=%s", request_id, bool(delivered))
            finally:
                PHOTO_SLOTS.release()

        t = threading.Thread(target=background_work, daemon=True)
        try:
            t.start()
        except Exception as exc:
            PHOTO_SLOTS.release()
            logger.error("photo_worker_failed request_id=%s kind=%s", request_id, type(exc).__name__)
            return kakao_text_response("사진 처리를 시작하지 못했습니다. 다시 보내 주십시오.")
        logger.info("kakao_response_ready mode=callback useCallback=true")
        return jsonify({"version": "2.0", "useCallback": True})


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

from __future__ import annotations

import base64
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

def extract_image_url(payload: Dict[str, Any]) -> Tuple[Optional[str], str]:
    action = _as_dict(payload.get("action"))
    detail_params = _as_dict(action.get("detailParams"))
    
    # 1. detailParams 검사
    for key, val_obj in detail_params.items():
        if isinstance(val_obj, dict):
            orig = val_obj.get("origin")
            if isinstance(orig, str) and orig.startswith("http"):
                return orig, f"detailParams.{key}.origin"
            val_str = val_obj.get("value")
            if isinstance(val_str, str) and val_str.startswith("http"):
                return val_str, f"detailParams.{key}.value"
            
    # 2. contexts / extra 검사
    contexts = _as_list(payload.get("contexts"))
    for ctx in contexts:
        ctx_d = _as_dict(ctx)
        params = _as_dict(ctx_d.get("params"))
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("http"):
                return v, f"contexts.params.{k}"
            elif isinstance(v, dict):
                sub_origin = v.get("origin") or v.get("value")
                if isinstance(sub_origin, str) and sub_origin.startswith("http"):
                    return sub_origin, f"contexts.params.{k}.sub"

    # 3. 사용자 발화 내 URL 직접 포함 검사
    user_request = _as_dict(payload.get("userRequest"))
    utterance = user_request.get("utterance", "")
    urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', utterance)
    if urls:
        return urls[0], "utterance_url"

    return None, "not_found"

def analyze_image_with_openai(image_url: str, user_text: str) -> str:
    if not OPENAI_API_KEY:
        return "⚠️ OpenAI API 키가 설정되어 있지 않습니다. Render 환경변수를 확인해주세요."

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    
    system_prompt = (
        "당신은 이재철 대표의 고전 철학과 홍익인간·재세이화 정신을 바탕으로 "
        "사용자가 올린 이미지와 질문에 대해 깊이 있고 명확하게(정명, 正名) 해설하는 AI 지음입니다."
    )
    
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text or "이 이미지를 고전 철학의 관점에서 깊이 있게 해설해 주세요."},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": OPENAI_IMAGE_DETAIL}}
                ]
            }
        ],
        "max_tokens": 1000
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=25)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            return f"⚠️ OpenAI 분석 중 오류가 발생했습니다. (상태코드: {response.status_code})"
    except Exception as e:
        logger.error(f"OpenAI API 호출 실패: {e}")
        return "⚠️ 이미지 분석 중 서버 오류가 발생했습니다."

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
        requests.post(callback_url, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"콜백 전송 실패: {e}")

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
    payload = request.get_json(silent=True) or {}
    user_request = _as_dict(payload.get("userRequest"))
    callback_url = user_request.get("callbackUrl")
    utterance = user_request.get("utterance", "")
    
    image_url, src_type = extract_image_url(payload)
    
    if not image_url:
        return jsonify({
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "사진을 찾지 못했습니다. 이미지를 포함하여 다시 전송해 주세요."
                        }
                    }
                ]
            }
        })

    # 동기 처리 시간 초과 방지를 위한 콜백 모드 분기
    if callback_url:
        def background_work():
            result_text = analyze_image_with_openai(image_url, utterance)
            send_callback(callback_url, result_text)
            
        t = threading.Thread(target=background_work, daemon=True)
        t.start()
        
        return jsonify({
            "version": "2.0",
            "useCallback": True,
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "🖼️ 이미지를 접수하여 깊이 있게 분석하고 있습니다. 잠시만 기다려 주세요..."
                        }
                    }
                ]
            }
        })
    else:
        result_text = analyze_image_with_openai(image_url, utterance)
        return jsonify({
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": result_text[:1000]
                        }
                    }
                ]
            }
        })

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

import base64
import json
import mimetypes
import os
import re
import threading
from urllib.parse import urlparse

import requests
from flask import Flask, jsonify, request
from openai import OpenAI

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna").strip()
PORT = int(os.getenv("PORT", "5000"))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

SYSTEM_PROMPT = """
너는 '바로몽땅찾아'의 친절한 AI 안내자 'AI지혜'다.
주 사용자는 60~70대도 포함하므로 쉬운 한국어로 짧고 또렷하게 답한다.

공통 원칙:
1. 사진에서 무엇이 보이는지 먼저 판단한다.
2. 확실한 것만 단정한다.
3. 애매하거나 흐린 부분은 '확인 필요'라고 표시한다.
4. 모르면 아는 척하지 않는다.
5. 사진이 흐리면 전체 사진 1장 + 확대 사진 1장을 다시 요청한다.
6. 답변은 가능한 한 700자 이내로 간결하게 한다.

한자·고문서·족보·비문·현판·옛문서는:
판독:
읽기:
뜻:
확인 필요:

일반 사물·식물·제품·안내문 등은:
무엇인지:
쉽게 설명:
주의/확인할 점:
""".strip()


def kakao_text(text: str):
    text = (text or "사진을 확인했지만 답변을 만들지 못했습니다.").strip()
    if len(text) > 950:
        text = text[:947] + "..."
    return {
        "version": "2.0",
        "template": {
            "outputs": [{"simpleText": {"text": text}}],
            "quickReplies": [
                {"messageText": "다른 사진 올리기", "action": "message", "label": "다른 사진"},
                {"messageText": "더 자세히", "action": "message", "label": "더 자세히"},
                {"messageText": "전문가 검토", "action": "message", "label": "전문가 검토"}
            ]
        }
    }


def callback_wait_response():
    return {
        "version": "2.0",
        "useCallback": True,
        "data": {"text": "사진을 받았습니다. AI지혜가 살펴보고 있습니다."}
    }


def normalize_urls(value):
    if not value:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if str(x).startswith(("http://", "https://"))]
    text = str(value).strip()
    if text.startswith("List(") and text.endswith(")"):
        text = text[5:-1]
    return re.findall(r'https?://[^\\s\\)\\]"\\']+', text)


def extract_secure_url(payload: dict):
    action = payload.get("action") or {}
    params = action.get("params") or {}
    detail_params = action.get("detailParams") or {}

    candidates = []
    if "secureimage" in params:
        candidates.append(params.get("secureimage"))

    secure_detail = detail_params.get("secureimage") or {}
    if isinstance(secure_detail, dict):
        candidates.append(secure_detail.get("value"))
        candidates.append(secure_detail.get("origin"))

    for raw in candidates:
        if not raw:
            continue
        obj = raw
        if isinstance(raw, str):
            try:
                obj = json.loads(raw)
            except Exception:
                pass

        if isinstance(obj, dict):
            urls = normalize_urls(obj.get("secureUrls"))
            if urls:
                return urls[0]

        if isinstance(obj, str):
            urls = normalize_urls(obj)
            if urls:
                return urls[0]

    return None


def download_image_as_data_url(image_url: str):
    parsed = urlparse(image_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("올바른 이미지 URL이 아닙니다.")

    r = requests.get(image_url, timeout=12)
    r.raise_for_status()

    content_type = r.headers.get("content-type", "").split(";")[0].strip()
    if not content_type.startswith("image/"):
        guessed, _ = mimetypes.guess_type(parsed.path)
        content_type = guessed or "image/jpeg"

    if len(r.content) > 15 * 1024 * 1024:
        raise ValueError("이미지 용량이 너무 큽니다.")

    encoded = base64.b64encode(r.content).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def analyze_image(image_url: str, user_text: str = ""):
    if client is None:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

    data_url = download_image_as_data_url(image_url)
    question = (user_text or "").strip()
    user_instruction = (
        f"사용자 질문: {question}\\n사진을 보고 위 질문에 답해줘."
        if question else
        "이 사진이 무엇인지 판독하고 사용자가 이해하기 쉽게 설명해줘."
    )

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=SYSTEM_PROMPT,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_instruction},
                {"type": "input_image", "image_url": data_url}
            ]
        }]
    )

    answer = getattr(response, "output_text", None)
    return (answer or "사진을 확인했지만 답변을 만들지 못했습니다. 사진을 다시 올려주세요.").strip()


def do_callback(callback_url: str, image_url: str, user_text: str):
    try:
        answer = analyze_image(image_url, user_text)
        body = kakao_text(answer)
    except Exception as e:
        app.logger.exception("AI 분석 실패")
        body = kakao_text(f"사진 분석 중 문제가 생겼습니다. 잠시 뒤 다시 올려주세요. ({type(e).__name__})")

    try:
        requests.post(callback_url, json=body, timeout=10).raise_for_status()
    except Exception:
        app.logger.exception("카카오 콜백 전송 실패")


@app.get("/")
def home():
    return "바로몽땅찾아 AI지혜 서버 정상 작동", 200


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "service": "바로몽땅찾아-AI지혜",
        "model": OPENAI_MODEL,
        "openai_key": bool(OPENAI_API_KEY)
    })


@app.post("/kakao/photo")
def kakao_photo():
    payload = request.get_json(silent=True) or {}
    image_url = extract_secure_url(payload)

    user_request = payload.get("userRequest") or {}
    utterance = user_request.get("utterance") or ""
    callback_url = user_request.get("callbackUrl")

    if not image_url:
        return jsonify(kakao_text(
            "사진 주소를 받지 못했습니다. 카카오 블록에서 "
            "@sys.plugin.secureimage 파라미터명 'secureimage'가 "
            "스킬로 전달되도록 설정했는지 확인해주세요."
        ))

    if callback_url:
        threading.Thread(
            target=do_callback,
            args=(callback_url, image_url, utterance),
            daemon=True
        ).start()
        return jsonify(callback_wait_response())

    try:
        answer = analyze_image(image_url, utterance)
        return jsonify(kakao_text(answer))
    except Exception as e:
        app.logger.exception("사진 분석 실패")
        return jsonify(kakao_text(
            f"사진 분석 중 문제가 생겼습니다. OPENAI_API_KEY와 서버 연결을 확인해주세요. ({type(e).__name__})"
        ))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
'''

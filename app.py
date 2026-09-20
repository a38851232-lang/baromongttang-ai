import base64
import json
import mimetypes
import os
import re
import urllib.parse
import urllib.request
from flask import Flask, jsonify, request
import openai
import threading

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
PORT = int(os.getenv("PORT", "10000"))

def kakao_text(text: str):
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": text
                    }
                }
            ]
        }
    }

def extract_image_url(payload: dict):
    action = payload.get("action") or {}
    params = action.get("params") or {}
    detail_params = action.get("detailParams") or {}
    
    candidates = []

    for v in params.values():
        candidates.append(v)

    for detail in detail_params.values():
        if isinstance(detail, dict):
            candidates.append(detail.get("value"))
            candidates.append(detail.get("origin"))
        else:
            candidates.append(detail)

    for c in candidates:
        if c is not None:
            text = str(c).strip()
            if text.startswith("List(") and text.endswith(")"):
                text = text[5:-1]
            found = re.findall(r'https?://[^\s\]\)"\']+', text)
            if found:
                return found[0]
            if text.startswith("http://") or text.startswith("https://"):
                return text

    return None

def encode_image_from_url(url: str):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        image_bytes = response.read()
        mime_type = response.headers.get_content_type()
        if not mime_type or mime_type == "application/octet-stream":
            mime_type, _ = mimetypes.guess_type(url)
            if not mime_type:
                mime_type = "image/jpeg"
        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{base64_data}"

def analyze_image(image_url: str, user_text: str = ""):
    if not OPENAI_API_KEY:
        return "오픈아이_API_KEY 설정이 되어있지 않습니다."

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    data_url = encode_image_from_url(image_url)

    prompt = (
        "당신은 고전 철학과 현판, 고문헌을 완벽하게 해독하는 최고의 인문학 AI 전문가입니다. "
        "제시된 이미지를 정밀하게 분석하여 다음 양식으로 답변해 주세요:\n\n"
        "1. 원문 판독 (한자 및 텍스트 원문)\n"
        "2. 현대적 의미 및 풀이 (철학적 배경과 깊이 있는 해석)\n"
        "3. 실생활 적용 및 교훈\n\n"
        f"사용자 추가 요청: {user_text}"
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content

def send_callback(callback_url: str, image_url: str, utterance: str):
    try:
        answer = analyze_image(image_url, utterance)
        payload = kakao_text(answer)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            callback_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        app.logger.exception("백그라운드 콜백 실패")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "바로몽 땡찾아-AI지혜",
        "model": OPENAI_MODEL,
        "openai_key": "설정됨" if OPENAI_API_KEY else "미설정"
    }), 200

@app.route("/kakao/photo", methods=["POST"])
def kakao_photo():
    payload = request.get_json(silent=True) or {}
    print("--- KAKAO PAYLOAD ---", json.dumps(payload, ensure_ascii=False))   
    image_url = extract_image_url(payload)
    
    user_request = payload.get("userRequest") or {}
    utterance = user_request.get("utterance") or ""
    callback_url = user_request.get("callbackUrl")

    if not image_url:
        return jsonify(
            kakao_text(
                "사진을 받지 못했습니다. "
                "카카오 블록의 필수 파라미터 이름을 secureimage로 하고, "
                "엔티티를 sys.plugin.secureimage로 설정했는지 확인해주세요."
            )
        )

    if callback_url:
        threading.Thread(
            target=send_callback,
            args=(callback_url, image_url, utterance),
            daemon=True
        ).start()

        return jsonify(
            kakao_text("현판 사진을 분석 중입니다. 잠시만 기다려 주세요.")
        )

    try:
        answer = analyze_image(image_url, utterance)
        return jsonify(kakao_text(answer))
    except Exception as exc:
        app.logger.exception("사진 분석 실패")
        return jsonify(
            kakao_text(
                f"사진은 서버까지 도착했으나 AI 분석 중 문제가 생겼습니다.\n"
                f"오류: {type(exc).__name__}"
            )
        ), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)

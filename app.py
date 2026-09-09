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
PORT = int(os.getenv("PORT", "10000"))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

SYSTEM_PROMPT = """
너는 '바로찾아 몽땅찾아'의 AI 안내자 'AI지혜'다.

답변 원칙:
1. 먼저 답한다.
2. 쉽게 풀어준다.
3. 근거를 말한다.
4. 불확실한 것은 숨기지 않는다.
5. 60~70대도 이해하기 쉬운 한국어로 짧고 또렷하게 쓴다.
6. 사진이 흐리면 전체 사진 1장과 확대 사진 1장을 다시 요청한다.
7. 모르면 아는 척하지 않는다.
8. 답변은 가급적 700자 이내로 한다.
[정확성 최우선 규칙]
- 한자·고문서·서화·현판·화제·낙관은 빠른 답보다 정확한 판독을 우선한다.
- 한 글자씩 확인한 뒤 전체 문맥과 그림 내용을 서로 대조한다.
- 유명 구절은 출전과 문맥까지 확인한다.
- 확신이 낮으면 절대 단정하지 말고 "정확한 판독을 위해 더 선명한 사진이 필요합니다"라고 답한다.
- 한자 답변은 가능하면 원문 → 음독 → 현토식 독법 → 뜻 → 출전 순서로 설명한다.
- 사용자가 오류를 지적하면 즉시 재검토하고 "잘못 판독했습니다. 정정합니다."라고 분명히 고친다.
- 예: 香遠益淸은 향원익청, "향기는 멀수록 더욱 맑다"로 판독한다.


한자·고문서·족보·비문·현판·옛문서이면:
판독:
읽기:
뜻:
확인 필요:

일반 사진이면:
찾았습니다. 이것은 ○○입니다.
쉽게 설명:
이렇게 판단한 이유:
확인 필요:
""".strip()


def kakao_text(text: str):
    text = (text or "사진을 확인했지만 답변을 만들지 못했습니다.").strip()

    if len(text) > 950:
        text = text[:947] + "..."

    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": text
                    }
                }
            ],
            "quickReplies": [
                {
                    "messageText": "다른 사진 올리기",
                    "action": "message",
                    "label": "다른 사진"
                },
                {
                    "messageText": "더 자세히",
                    "action": "message",
                    "label": "더 자세히"
                },
                {
                    "messageText": "전문가 검토",
                    "action": "message",
                    "label": "전문가 검토"
                }
            ]
        }
    }


def wait_callback_response():
    return {
        "version": "2.0",
        "useCallback": True,
        "data": {
            "text": "사진을 받았습니다. AI지혜가 살펴보고 있습니다."
        }
    }


def normalize_urls(value):
    if not value:
        return []

    if isinstance(value, list):
        return [
            str(x)
            for x in value
            if str(x).startswith(("http://", "https://"))
        ]

    text = str(value).strip()

    if text.startswith("List(") and text.endswith(")"):
        text = text[5:-1]

    return re.findall(r'https?://[^\s\)\]"\']+', text)


def extract_image_url(payload: dict):
    action = payload.get("action") or {}
    params = action.get("params") or {}
    detail_params = action.get("detailParams") or {}

    candidates = []

    if "secureimage" in params:
        candidates.append(params.get("secureimage"))

    detail = detail_params.get("secureimage") or {}

    if isinstance(detail, dict):
        candidates.append(detail.get("value"))
        candidates.append(detail.get("origin"))

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


def image_url_to_data_url(image_url: str):
    parsed = urlparse(image_url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError("올바른 이미지 URL이 아닙니다.")

    response = requests.get(image_url, timeout=15)
    response.raise_for_status()

    content_type = response.headers.get(
        "content-type",
        ""
    ).split(";")[0].strip()

    if not content_type.startswith("image/"):
        guessed, _ = mimetypes.guess_type(parsed.path)
        content_type = guessed or "image/jpeg"

    if len(response.content) > 15 * 1024 * 1024:
        raise ValueError("이미지 용량이 너무 큽니다.")

    encoded = base64.b64encode(response.content).decode("ascii")

    return f"data:{content_type};base64,{encoded}"


def analyze_image(image_url: str, user_text: str = ""):
    if client is None:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")

    data_url = image_url_to_data_url(image_url)

    question = (user_text or "").strip()

    if question:
        prompt = (
            f"사용자 질문: {question}\n"
            "사진을 보고 위 질문에 답해줘."
        )
    else:
        prompt = (
            "사진이 무엇인지 판독하고 "
            "사용자가 바로 이해할 수 있게 설명해줘."
        )

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image_url": data_url
                    }
                ]
            }
        ]
    )

    answer = getattr(response, "output_text", None)

    if not answer:
        return (
            "사진을 확인했지만 정확한 답변을 만들지 못했습니다. "
            "사진을 조금 더 선명하게 다시 올려주세요."
        )

    return answer.strip()


def send_callback(callback_url: str, image_url: str, utterance: str):
    try:
        answer = analyze_image(image_url, utterance)
        body = kakao_text(answer)

    except Exception as exc:
        app.logger.exception("AI 분석 실패")

        body = kakao_text(
            "사진 분석 중 문제가 생겼습니다. "
            "잠시 뒤 사진을 다시 올려주세요. "
            f"({type(exc).__name__})"
        )

    try:
        requests.post(
            callback_url,
            json=body,
            timeout=10
        ).raise_for_status()

    except Exception:
        app.logger.exception("카카오 콜백 전송 실패")


@app.get("/")
def home():
    return "바로찾아 몽땅찾아 서버 정상 작동 중<br>운영사: 태교 에이아이 주식회사", 200


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "바로몽땅찾아-AI지혜",
            "model": OPENAI_MODEL,
            "openai_key": bool(OPENAI_API_KEY)
        }
    ), 200


@app.post("/photo")
def photo():
    return kakao_photo()


@app.post("/kakao/photo")
def kakao_photo():
    payload = request.get_json(silent=True) or {}

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

        return jsonify(wait_callback_response())

    try:
        answer = analyze_image(image_url, utterance)
        return jsonify(kakao_text(answer))

    except Exception as exc:
        app.logger.exception("사진 분석 실패")

        return jsonify(
            kakao_text(
                "사진은 서버까지 도착했지만 AI 분석 중 문제가 생겼습니다. "
                "Render의 OPENAI_API_KEY 설정을 확인해주세요. "
                f"({type(exc).__name__})"
            )
        )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False
    )

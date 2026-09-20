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

try:  # .env 파일을 쓰는 경우 (없어도 정상 동작)
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

try:  # openai>=1.0
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# 1. 설정
# ---------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# 기본 모델(비전 지원 모델). 필요하면 환경변수로 교체한다.
#   예) gpt-5.4-mini / gpt-4.1-mini / gpt-4o-mini 등 이미지 입력을 지원하는 모델
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5.4-mini").strip()
OPENAI_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv("OPENAI_FALLBACK_MODELS", "gpt-4.1-mini,gpt-4o-mini").split(",")
    if m.strip()
]

# low / high / auto (모델이 지원하면 original)
OPENAI_IMAGE_DETAIL = os.getenv("OPENAI_IMAGE_DETAIL", "auto").strip() or "auto"
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "700"))
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "45"))

IMAGE_DOWNLOAD_TIMEOUT = float(os.getenv("IMAGE_DOWNLOAD_TIMEOUT", "10"))
MAX_IMAGE_COUNT = int(os.getenv("MAX_IMAGE_COUNT", "3"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))  # 이미지 1장당 15MB

# 카카오 응답 제약
KAKAO_TEXT_LIMIT = 1000  # simpleText 한 개당 최대 길이
KAKAO_MAX_OUTPUTS = 3  # 한 응답에 넣을 수 있는 output 개수
KAKAO_CALLBACK_TIMEOUT = float(os.getenv("KAKAO_CALLBACK_TIMEOUT", "10"))

# 콜백 사용 여부(오픈빌더 블록에서 "콜백 사용"을 켠 경우에만 동작한다)
USE_CALLBACK = os.getenv("USE_CALLBACK", "false").strip().lower() in ("1", "true", "yes", "y", "on")

# 사용자 발화를 프롬프트로 쓸지 여부(블록 발화가 그대로 들어오는 경우가 많아 기본 false)
USE_UTTERANCE_AS_PROMPT = os.getenv("USE_UTTERANCE_AS_PROMPT", "false").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)

DEFAULT_QUESTION = "이 이미지에 무엇이 있는지 한국어로 자세히 설명해 주세요."
SYSTEM_PROMPT = (
    "너는 카카오톡 챗봇의 이미지 분석 도우미다. "
    "사용자가 보낸 이미지를 보고 한국어로 정확하고 간결하게 답한다. "
    "규칙: (1) 핵심부터 말하고 필요하면 3~5개 항목으로 나눠 정리한다. "
    "(2) 마크다운 기호(#, *, **)는 쓰지 않는다. 항목 기호는 '•' 를 쓴다. "
    "(3) 이미지에서 확인할 수 없는 내용은 추측하지 말고 '이미지에서 확인할 수 없습니다'라고 말한다. "
    "(4) 전체 답변은 600자 이내로 작성한다."
)

# 발화 키워드별 프롬프트 라우팅
PROMPT_RULES: List[Tuple[Tuple[str, ...], str]] = [
    (
        ("글자", "텍스트", "ocr", "읽어", "추출"),
        "이 이미지에 보이는 모든 글자를 원본 그대로 옮겨 적어 주세요. "
        "표가 있으면 항목별로 정리해 주세요. 이미지에 글자가 없으면 '글자가 없습니다'라고 답하세요.",
    ),
    (
        ("번역", "해석", "translate"),
        "이 이미지에 있는 글자를 원문 그대로 적고, 바로 아래에 한국어 번역을 함께 적어 주세요.",
    ),
    (
        ("영수증", "결제", "금액", "가격"),
        "이 영수증(또는 결제 내역) 이미지에서 상호명, 날짜, 결제 금액, 결제 수단, 주요 품목을 뽑아 정리해 주세요. "
        "확인할 수 없는 항목은 '확인 불가'라고 표시하세요.",
    ),
    (
        ("음식", "메뉴", "먹", "요리"),
        "이 이미지에 보이는 음식의 이름과 특징, 대략적인 재료를 한국어로 알려 주세요.",
    ),
]

QUICK_REPLIES = [
    {"action": "message", "label": "글자만 추출", "messageText": "글자만 추출해줘"},
    {"action": "message", "label": "다시 설명", "messageText": "이미지 다시 설명해줘"},
]


# ---------------------------------------------------------------------------
# 2. Flask / 로깅 초기화
# ---------------------------------------------------------------------------
app = Flask(__name__)
try:  # Flask 2.3+ : 한글 응답을 그대로 출력
    app.json.ensure_ascii = False
except Exception:  # pragma: no cover
    app.config["JSON_AS_ASCII"] = False

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("kakao-vision-bot")

client = None
if OpenAI is not None and OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT)


# ---------------------------------------------------------------------------
# 3. 이미지 URL 추출 유틸
# ---------------------------------------------------------------------------
URL_PATTERN = re.compile(r"https?://[^\s\"'<>()\[\]{},]+", re.IGNORECASE)
LIST_WRAPPER_PATTERN = re.compile(r"^List\s*\((.*)\)$", re.IGNORECASE | re.DOTALL)
IMAGE_EXT_PATTERN = re.compile(r"\.(png|jpe?g|gif|webp|bmp|heic|heif|tiff?)(?:$|[?#])", re.IGNORECASE)

# 키 이름에 아래 문자열이 들어 있으면 "이미지 파라미터"로 간주한다.
HINTED_PARAM_KEYS = (
    "image",
    "img",
    "photo",
    "picture",
    "media",
    "secure",
    "attach",
    "file",
)

# 확장자가 없어도 이미지로 볼 수 있는 URL 힌트
IMAGE_URL_HINTS = ("secure", "image", "img", "photo", "picture", "thumb", "kakao", "cdn", "blob")

MAX_SCAN_DEPTH = 8


def _clean_url(value: Any) -> str:
    """URL 문자열에서 따옴표/꼬리 기호를 제거한다."""
    if not isinstance(value, str):
        return ""
    url = value.strip().strip("\"'")
    url = url.rstrip(".,;:)]}>\\")
    return url


def _is_http_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _append_unique(target: List[str], value: Any) -> None:
    """중복 없이 URL 목록에 추가한다."""
    url = _clean_url(value)
    if _is_http_url(url) and url not in target:
        target.append(url)


def _strip_list_wrapper(value: str) -> str:
    """카카오가 문자열로 감싸 보내는 'List(http://a, http://b)' 형태를 벗긴다."""
    text = (value or "").strip()
    match = LIST_WRAPPER_PATTERN.match(text)
    if match:
        return match.group(1).strip()
    return text


def _try_json(value: str) -> Optional[Any]:
    """문자열이 JSON(dict/list)이면 파싱해서 돌려준다."""
    text = (value or "").strip()
    if not text or text[0] not in "[{":
        return None
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, (dict, list)) else None


def _extract_urls_from_text(value: str) -> List[str]:
    """문자열 하나에서 URL 후보를 모두 뽑아낸다."""
    urls: List[str] = []
    text = _strip_list_wrapper(value)
    if not text:
        return urls
    for raw in URL_PATTERN.findall(text):
        url = _clean_url(raw)
        if _is_http_url(url) and url not in urls:
            urls.append(url)
    return urls


def _is_probably_image_url(url: str) -> bool:
    """확장자가 없어도 이미지일 가능성이 높은 URL 인지 판단한다(최후 수단용)."""
    if not _is_http_url(url):
        return False
    if IMAGE_EXT_PATTERN.search(url):
        return True
    lowered = url.lower()
    return any(hint in lowered for hint in IMAGE_URL_HINTS)


def _urls_from_any(value: Any) -> List[str]:
    """dict / list / JSON 문자열 / 문자열 어느 형태여도 URL 을 뽑아낸다."""
    urls: List[str] = []

    if isinstance(value, str):
        inner = _try_json(value)
        if inner is not None:
            return _urls_from_any(inner)
        return _extract_urls_from_text(value)

    if isinstance(value, dict):
        # 자주 쓰이는 키를 먼저 본다.
        for key in ("secureUrls", "secure_urls", "value", "origin", "url", "imageUrl", "image_url"):
            if key in value:
                for url in _urls_from_any(value[key]):
                    _append_unique(urls, url)
        if not urls:
            for item in value.values():
                for url in _urls_from_any(item):
                    _append_unique(urls, url)
        return urls

    if isinstance(value, (list, tuple, set)):
        for item in value:
            for url in _urls_from_any(item):
                _append_unique(urls, url)
        return urls

    return urls


def _scan_for_urls(
    node: Any,
    require_image_key: bool,
    hinted: bool = False,
    depth: int = 0,
) -> List[str]:
    """
    payload 전체를 재귀 탐색하며 URL 을 수집한다.
      require_image_key=True  : 'image', 'secureimage' 등 이미지성 키 아래의 URL 만 채택
      require_image_key=False : 키와 무관하게 이미지로 보이는 URL 채택
    """
    urls: List[str] = []

    if depth > MAX_SCAN_DEPTH or node is None:
        return urls

    if isinstance(node, dict):
        for key, value in node.items():
            key_lower = str(key).lower()
            child_hinted = hinted or any(hint in key_lower for hint in HINTED_PARAM_KEYS)
            for url in _scan_for_urls(value, require_image_key, child_hinted, depth + 1):
                _append_unique(urls, url)
        return urls

    if isinstance(node, (list, tuple, set)):
        for item in node:
            for url in _scan_for_urls(item, require_image_key, hinted, depth + 1):
                _append_unique(urls, url)
        return urls

    if isinstance(node, str):
        inner = _try_json(node)
        if inner is not None:
            return _scan_for_urls(inner, require_image_key, hinted, depth + 1)
        if require_image_key and not hinted:
            return urls
        for url in _extract_urls_from_text(node):
            if require_image_key or _is_probably_image_url(url):
                _append_unique(urls, url)
        return urls

    return urls


def extract_image_url(payload: Dict[str, Any]) -> List[str]:
    """
    카카오 오픈빌더 스킬 요청 payload 에서 이미지 URL 목록을 추출한다.

    지원하는 위치
      1) 일반 이미지 전송   : userRequest.params.media.url
      2) 이미지 보안전송 플러그인 :
           action.detailParams.<이미지파라미터>.value  (JSON 문자열, secureUrls 포함)
           action.detailParams.<이미지파라미터>.origin ('List(http://..., http://...)' 형태)
           action.params.<이미지파라미터>
           userRequest.params.<이미지파라미터>
      3) 그 밖의 위치     : payload 전체 재귀 탐색(이미지성 키 우선 → 이미지로 보이는 URL)
    """
    found: List[str] = []

    if not isinstance(payload, dict):
        return found

    action = payload.get("action") or {}
    user_request = payload.get("userRequest") or {}
    if not isinstance(action, dict):
        action = {}
    if not isinstance(user_request, dict):
        user_request = {}

    # (1) 일반 이미지 전송 : userRequest.params.media
    media = (user_request.get("params") or {}).get("media") if isinstance(
        user_request.get("params"), dict
    ) else None
    if isinstance(media, dict):
        media_type = str(media.get("type", "image")).lower()
        if media_type in ("image", "photo", "picture", "img", ""):
            for url in _urls_from_any(media.get("url")):
                _append_unique(found, url)
    elif media:
        for url in _urls_from_any(media):
            _append_unique(found, url)

    # (2) detailParams / params 의 이미지성 파라미터
    for source in (
        action.get("detailParams"),
        action.get("params"),
        user_request.get("params"),
    ):
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            key_lower = str(key).lower()
            if any(hint in key_lower for hint in HINTED_PARAM_KEYS):
                for url in _urls_from_any(value):
                    _append_unique(found, url)

    # (3) 1차 넓은 탐색 : 이미지성 키 아래의 URL 만
    if not found:
        for url in _scan_for_urls(payload, require_image_key=True):
            _append_unique(found, url)

    # (4) 2차 넓은 탐색 : 키와 무관하게 이미지로 보이는 URL
    if not found:
        for url in _scan_for_urls(payload, require_image_key=False):
            _append_unique(found, url)

    return found


# ---------------------------------------------------------------------------
# 4. 이미지 다운로드 / OpenAI Vision 분석
# ---------------------------------------------------------------------------
SIGNATURES: List[Tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
]


def _sniff_mime(data: bytes, header_value: Optional[str]) -> str:
    """파일 시그니처 → Content-Type 헤더 순으로 MIME 타입을 판별한다."""
    if data[:12].startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    for signature, mime in SIGNATURES:
        if data.startswith(signature):
            return mime
    if header_value:
        mime = header_value.split(";")[0].strip().lower()
        if mime.startswith("image/"):
            return mime
    return "image/jpeg"


def download_image(url: str) -> Optional[Tuple[bytes, str]]:
    """
    이미지 바이트를 내려받아 (bytes, mime) 로 돌려준다.
    카카오 보안전송 URL 은 확장자가 없는 경우가 많고, OpenAI 가 URL 을 직접
    가져오지 못하는 경우가 있어 base64(data URL)로 변환해 전달하기 위해 사용한다.
    실패하면 None 을 돌려주고, 호출부에서 URL 직접 전달로 폴백한다.
    """
    try:
        response = requests.get(
            url,
            timeout=IMAGE_DOWNLOAD_TIMEOUT,
            stream=True,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; KakaoVisionBot/1.0)",
                "Accept": "image/*,*/*;q=0.8",
            },
        )
        response.raise_for_status()

        chunks: List[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                logger.warning("이미지 용량 초과로 건너뜀: %s", _mask_url(url))
                return None
            chunks.append(chunk)

        data = b"".join(chunks)
        if not data:
            logger.warning("이미지 응답이 비어 있음: %s", _mask_url(url))
            return None
        return data, _sniff_mime(data, response.headers.get("Content-Type"))
    except Exception as exc:
        logger.warning("이미지 다운로드 실패(%s): %s", _mask_url(url), exc)
        return None


def _to_data_url(data: bytes, mime: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _mask_url(url: str) -> str:
    """로그에 그대로 남기면 위험한 토큰 쿼리를 가린다."""
    try:
        parts = urlsplit(url)
        return f"{parts.scheme}://{parts.netloc}{parts.path}?<masked>"
    except Exception:
        return "<invalid-url>"


def _build_messages(image_urls: List[str], question: str) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": question}]

    for url in image_urls[:MAX_IMAGE_COUNT]:
        downloaded = download_image(url)
        if downloaded:
            data, mime = downloaded
            image_entry = {
                "type": "image_url",
                "image_url": {"url": _to_data_url(data, mime), "detail": OPENAI_IMAGE_DETAIL},
            }
        else:
            # 다운로드 실패 시 URL 을 그대로 넘겨 OpenAI 가 직접 가져오도록 시도
            image_entry = {
                "type": "image_url",
                "image_url": {"url": url, "detail": OPENAI_IMAGE_DETAIL},
            }
        content.append(image_entry)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _model_candidates() -> List[str]:
    models: List[str] = []
    for model in [OPENAI_VISION_MODEL] + OPENAI_FALLBACK_MODELS:
        if model and model not in models:
            models.append(model)
    return models


def analyze_with_openai(image_urls: List[str], question: str) -> str:
    """
    이미지 목록을 OpenAI Vision API 로 분석해 텍스트를 돌려준다.
    모델/파라미터 호환 문제에 대비해 후보 모델 → detail 값 순으로 재시도한다.
    """
    if client is None:
        raise RuntimeError("OPENAI_API_KEY 가 설정되지 않았습니다.")

    messages = _build_messages(image_urls, question)
    errors: List[str] = []

    for model in _model_candidates():
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_completion_tokens=OPENAI_MAX_OUTPUT_TOKENS,
            )
            text = ""
            if response.choices:
                text = (response.choices[0].message.content or "").strip()
            if text:
                logger.info("OpenAI 분석 성공 (model=%s, images=%d)", model, len(image_urls))
                return text
            errors.append(f"{model}: 빈 응답")
        except Exception as exc:
            message = str(exc)
            errors.append(f"{model}: {message}")
            logger.warning("OpenAI 호출 실패 (model=%s): %s", model, message)

            # detail 값 미지원 모델 대응 → auto 로 1회 재시도
            if "detail" in message.lower():
                try:
                    retry_messages = json.loads(json.dumps(messages))
                    for message_item in retry_messages[1]["content"]:
                        if message_item.get("type") == "image_url":
                            message_item["image_url"]["detail"] = "auto"
                    response = client.chat.completions.create(
                        model=model,
                        messages=retry_messages,  # type: ignore[arg-type]
                        max_completion_tokens=OPENAI_MAX_OUTPUT_TOKENS,
                    )
                    text = (response.choices[0].message.content or "").strip()
                    if text:
                        logger.info("OpenAI 분석 성공 (model=%s, detail=auto)", model)
                        return text
                except Exception as retry_exc:  # pragma: no cover
                    logger.warning("detail=auto 재시도 실패: %s", retry_exc)

    raise RuntimeError(" / ".join(errors) if errors else "OpenAI 호출에 실패했습니다.")


# ---------------------------------------------------------------------------
# 5. 카카오 스킬 응답 생성
# ---------------------------------------------------------------------------
def _split_text(text: str, limit: int = KAKAO_TEXT_LIMIT, max_chunks: int = KAKAO_MAX_OUTPUTS) -> List[str]:
    """카카오 simpleText 길이 제한(1000자)에 맞춰 문단 단위로 나눈다."""
    text = (text or "").strip()
    if not text:
        return ["이미지를 분석하지 못했습니다. 잠시 후 다시 시도해 주세요."]

    chunks: List[str] = []
    buffer = ""
    truncated = False

    for block in text.split("\n"):
        block = block.rstrip()
        candidate = f"{buffer}\n{block}" if buffer else block

        if len(candidate) <= limit:
            buffer = candidate
            continue

        if buffer:
            chunks.append(buffer)
            buffer = ""

        if len(chunks) >= max_chunks:
            truncated = True
            break

        while len(block) > limit:
            chunks.append(block[:limit])
            block = block[limit:]
            if len(chunks) >= max_chunks:
                truncated = True
                break

        if truncated:
            break

        buffer = block

    if not truncated and buffer and len(chunks) < max_chunks:
        chunks.append(buffer)

    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
        truncated = True

    if truncated and chunks:
        room = max(limit - 2, 1)
        last = chunks[-1]
        chunks[-1] = (last[:room] if len(last) > room else last) + " …"

    return chunks or ["이미지를 분석하지 못했습니다. 잠시 후 다시 시도해 주세요."]


def build_kakao_response(
    text: str,
    quick_replies: Optional[List[Dict[str, str]]] = None,
    image_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    카카오 오픈빌더 스킬 응답 규격(version 2.0, template.outputs)을 만든다.
      - outputs: simpleText 최대 3개 (+ 선택적 simpleImage)
      - quickReplies: 최대 10개
    """
    outputs: List[Dict[str, Any]] = []

    if image_url:
        outputs.append({"simpleImage": {"imageUrl": image_url, "altText": "분석한 이미지"}})

    for chunk in _split_text(text):
        if len(outputs) >= KAKAO_MAX_OUTPUTS:
            break
        outputs.append({"simpleText": {"text": chunk}})

    if not outputs:
        outputs.append({"simpleText": {"text": "이미지를 분석하지 못했습니다."}})

    template: Dict[str, Any] = {"outputs": outputs}
    if quick_replies:
        template["quickReplies"] = quick_replies[:10]

    return {"version": "2.0", "template": template}


def build_callback_ack() -> Dict[str, Any]:
    """콜백 사용 시 1차로 즉시 돌려주는 응답."""
    return {"version": "2.0", "useCallback": True, "data": {"status": "processing"}}


def send_callback(callback_url: str, body: Dict[str, Any]) -> None:
    """분석 결과를 카카오 callbackUrl 로 전송한다."""
    try:
        response = requests.post(
            callback_url,
            json=body,
            timeout=KAKAO_CALLBACK_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        logger.info("콜백 전송 완료 (status=%s)", response.status_code)
    except Exception as exc:
        logger.error("콜백 전송 실패: %s", exc)


# ---------------------------------------------------------------------------
# 6. 프롬프트 결정
# ---------------------------------------------------------------------------
def _pick_question(payload: Dict[str, Any]) -> str:
    """사용자 발화/블록 정보를 보고 분석 프롬프트를 정한다."""
    utterance = ""
    user_request = payload.get("userRequest")
    if isinstance(user_request, dict):
        utterance = str(user_request.get("utterance") or "").strip()

    if utterance:
        lowered = utterance.lower()
        for keywords, prompt in PROMPT_RULES:
            if any(keyword in lowered for keyword in keywords):
                return prompt
        if USE_UTTERANCE_AS_PROMPT and len(utterance) > 1:
            return utterance

    return DEFAULT_QUESTION


# ---------------------------------------------------------------------------
# 7. 라우팅
# ---------------------------------------------------------------------------
def _process_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """payload → 이미지 추출 → OpenAI 분석 → 카카오 응답."""
    image_urls = extract_image_url(payload)
    logger.info("이미지 URL %d개 추출: %s", len(image_urls), [_mask_url(u) for u in image_urls])

    if not image_urls:
        return build_kakao_response(
            "이미지를 찾지 못했습니다.\n"
            "사진을 한 장 보내주시거나, 블록의 필수 파라미터에 이미지 플러그인"
            "(사진 보안 전송)을 설정했는지 확인해 주세요.",
            quick_replies=QUICK_REPLIES,
        )

    question = _pick_question(payload)

    try:
        answer = analyze_with_openai(image_urls, question)
    except Exception as exc:
        logger.error("이미지 분석 실패: %s", exc)
        return build_kakao_response(
            "이미지를 분석하지 못했습니다. 잠시 후 다시 시도해 주세요.\n"
            "같은 문제가 계속되면 관리자에게 문의해 주세요.",
            quick_replies=QUICK_REPLIES,
        )

    return build_kakao_response(answer, quick_replies=QUICK_REPLIES)


@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health() -> Any:
    """헬스 체크(스킬 등록 전 접속 확인용)."""
    return jsonify(
        {
            "status": "ok",
            "openai_key": bool(OPENAI_API_KEY),
            "model": OPENAI_VISION_MODEL,
            "callback_mode": USE_CALLBACK,
        }
    )


@app.route("/webhook", methods=["POST"])
def webhook() -> Any:
    """
    카카오 오픈빌더 스킬 엔드포인트.
    어떤 경우에도 카카오 규격(HTTP 200 + version/template)으로 응답한다.
    """
    payload: Dict[str, Any] = request.get_json(silent=True) or {}

    if not isinstance(payload, dict) or not payload:
        logger.warning("payload 파싱 실패 또는 비어 있음")
        return jsonify(build_kakao_response("요청을 이해하지 못했습니다. 다시 시도해 주세요."))

    try:
        callback_url = ""
        user_request = payload.get("userRequest")
        if isinstance(user_request, dict):
            callback_url = str(user_request.get("callbackUrl") or "").strip()

        if USE_CALLBACK and callback_url:
            # 즉시 응답 후 백그라운드에서 분석 → 결과를 callbackUrl 로 전송
            def worker() -> None:
                try:
                    result = _process_payload(payload)
                except Exception as exc:  # pragma: no cover
                    logger.error("콜백 처리 중 예외: %s", exc)
                    result = build_kakao_response(
                        "이미지를 분석하지 못했습니다. 잠시 후 다시 시도해 주세요."
                    )
                send_callback(callback_url, result)

            threading.Thread(target=worker, daemon=True).start()
            return jsonify(build_callback_ack())

        return jsonify(_process_payload(payload))

    except Exception as exc:  # 어떤 예외가 나도 카카오 규격으로 응답
        logger.exception("webhook 처리 중 예외: %s", exc)
        return jsonify(
            build_kakao_response("일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY 가 비어 있습니다. .env 또는 환경변수를 설정하세요.")
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")

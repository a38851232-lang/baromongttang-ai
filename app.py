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
        logger.error("image_fetch_failed exception=%s", type(exc).__name__)
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
                logger.error("openai_failed empty_content")
                return "⚠️ 이미지 분석 답변이 비어 있습니다. 잠시 후 다시 시도해 주세요."
            return content.strip()

        logger.error("openai_failed http=%s", response.status_code)
        return f"⚠️ OpenAI 분석 중 오류가 발생했습니다. (상태코드: {response.status_code})"
    except Exception as exc:
        logger.error("openai_failed exception=%s", type(exc).__name__)
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
    callback_url = (
        callback_url
        if isinstance(callback_url, str) and callback_url.startswith("https://")
        else None
    )

    utterance = user_request.get("utterance")
    utterance = utterance if isinstance(utterance, str) else ""

    image_url, src_type = extract_image_url(payload)
    logger.info(
        "photo_request source=%s image_found=%s callback_present=%s",
        src_type,
        bool(image_url),
        bool(callback_url),
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

    # Prefer Kakao callback when callbackUrl is supplied.
    if callback_url:
        def background_work():
            result_text = analyze_image_with_openai(image_url, utterance)
            send_callback(callback_url, result_text)

        t = threading.Thread(target=background_work, daemon=True)
        t.start()
        return jsonify({"version": "2.0", "useCallback": True})

    # Preserve synchronous behavior when callbackUrl is unavailable.
    result_text = analyze_image_with_openai(image_url, utterance)
    return kakao_text_response(result_text)


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
        logger.error("image_fetch_failed exception=%s", type(exc).__name__)
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
                logger.error("openai_failed empty_content")
                return "⚠️ 이미지 분석 답변이 비어 있습니다. 잠시 후 다시 시도해 주세요."
            return content.strip()

        logger.error("openai_failed http=%s", response.status_code)
        return f"⚠️ OpenAI 분석 중 오류가 발생했습니다. (상태코드: {response.status_code})"
    except Exception as exc:
        logger.error("openai_failed exception=%s", type(exc).__name__)
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
    callback_url = (
        callback_url
        if isinstance(callback_url, str) and callback_url.startswith("https://")
        else None
    )

    utterance = user_request.get("utterance")
    utterance = utterance if isinstance(utterance, str) else ""

    image_url, src_type = extract_image_url(payload)
    logger.info(
        "photo_request source=%s image_found=%s callback_present=%s",
        src_type,
        bool(image_url),
        bool(callback_url),
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

    # Prefer Kakao callback when callbackUrl is supplied.
    if callback_url:
        def background_work():
            result_text = analyze_image_with_openai(image_url, utterance)
            send_callback(callback_url, result_text)

        t = threading.Thread(target=background_work, daemon=True)
        t.start()
        return jsonify({"version": "2.0", "useCallback": True})

    # Preserve synchronous behavior when callbackUrl is unavailable.
    result_text = analyze_image_with_openai(image_url, utterance)
    return kakao_text_response(result_text)


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
        logger.error("image_fetch_failed exception=%s", type(exc).__name__)
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
                logger.error("openai_failed empty_content")
                return "⚠️ 이미지 분석 답변이 비어 있습니다. 잠시 후 다시 시도해 주세요."
            return content.strip()

        logger.error("openai_failed http=%s", response.status_code)
        return f"⚠️ OpenAI 분석 중 오류가 발생했습니다. (상태코드: {response.status_code})"
    except Exception as exc:
        logger.error("openai_failed exception=%s", type(exc).__name__)
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
    callback_url = (
        callback_url
        if isinstance(callback_url, str) and callback_url.startswith("https://")
        else None
    )

    utterance = user_request.get("utterance")
    utterance = utterance if isinstance(utterance, str) else ""

    image_url, src_type = extract_image_url(payload)
    logger.info(
        "photo_request source=%s image_found=%s callback_present=%s",
        src_type,
        bool(image_url),
        bool(callback_url),
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

    # Prefer Kakao callback when callbackUrl is supplied.
    if callback_url:
        def background_work():
            result_text = analyze_image_with_openai(image_url, utterance)
            send_callback(callback_url, result_text)

        t = threading.Thread(target=background_work, daemon=True)
        t.start()
        return jsonify({"version": "2.0", "useCallback": True})

    # Preserve synchronous behavior when callbackUrl is unavailable.
    result_text = analyze_image_with_openai(image_url, utterance)
    return kakao_text_response(result_text)


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

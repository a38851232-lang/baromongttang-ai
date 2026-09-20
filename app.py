def _intent_label(payload: Dict[str, Any]) -> str:
    intent = _as_dict(payload.get("intent"))
    name = str(intent.get("name") or "-")
    extra = _as_dict(intent.get("extra"))
    reason = _as_dict(extra.get("reason"))
    message = reason.get("message")
    return "%s (%s)" % (name, message) if message else name


@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
@app.route("/healthz", methods=["GET"])
def health() -> Any:
    """헬스 체크 + 현재 설정 확인. ?ping=1&token=... 이면 모델을 실제로 한 번씩 호출한다."""
    report: Dict[str, Any] = {
        "status": "ok",
        "version": VERSION,
        "openai_key_present": bool(OPENAI_API_KEY),
        "vision_models": _model_candidates(),
        "image_detail": OPENAI_IMAGE_DETAIL,
        "callback_mode": CALLBACK_MODE,
        "callback_ack_delay": CALLBACK_ACK_DELAY,
        "callback_ack_text": CALLBACK_ACK_TEXT,
        "sync_budget": SYNC_BUDGET,
        "pillow": _HAS_PIL,
        "debug_endpoint": bool(DEBUG_TOKEN),
        "keepalive": bool(KEEPALIVE_URL),
        "storage": (store.info() if store else {"backend": "none", "enabled": False}),
        "storage_upload": STORAGE_UPLOAD,
        "storage_read_cache": STORAGE_READ_CACHE,
        "storage_expire_safety": STORAGE_EXPIRE_SAFETY,
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if request.args.get("ping") == "1":
        if not DEBUG_TOKEN or request.args.get("token") != DEBUG_TOKEN:
            return jsonify({"error": "unauthorized"}), 401
        ping_results: Dict[str, str] = {}
        deadline = time.time() + 30
        for model in _model_candidates():
            if time.time() > deadline:
                ping_results[model] = "건너뜀(시간 초과)"
                continue
            try:
                response = client.chat.completions.create(  # type: ignore[union-attr]
                    model=model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_completion_tokens=5,
                )
                ping_results[model] = "OK" if response.choices else "빈 응답"
            except Exception as exc:
                ping_results[model] = "실패: %s" % str(exc)[:200]
        report["model_ping"] = ping_results

    return jsonify(report)


@app.route("/webhook", methods=["POST"])
@app.route("/kakao", methods=["POST"])
def webhook() -> Any:
    """
    카카오 오픈빌더 스킬 엔드포인트.
    어떤 경우에도 카카오 규격(HTTP 200 + version/...)으로 응답한다.
    """
    payload = _read_payload()
    if not payload:
        logger.warning("payload 파싱 실패 또는 비어 있음")
        return jsonify(build_kakao_response("요청을 이해하지 못했습니다. 다시 시도해 주세요."))

    if LOG_PAYLOAD:
        try:
            logger.info("PAYLOAD: %s", json.dumps(payload, ensure_ascii=False)[:6000])
        except Exception:
            logger.info("PAYLOAD(직렬화 실패): %r", payload)

    try:
        callback_url = str(_as_dict(payload.get("userRequest")).get("callbackUrl") or "").strip()
        logger.info(
            "요청 수신: intent=%s, callback=%s, mode=%s",
            _intent_label(payload), "있음" if callback_url else "없음", CALLBACK_MODE,
        )

        # (a) 콜백 비활성 → 동기 (예산 초과 시 우리 문구로 마감)
        if CALLBACK_MODE == "false":
            return jsonify(_process_payload_sync_budget(payload))

        # (b) 콜백 모드인데 callbackUrl 이 없음 → AI 챗봇 전환/블록 설정 문제. 동기로 처리.
        if not callback_url:
            logger.warning(
                "callbackUrl 이 없습니다 → 동기 처리. "
                "블록의 'Callback 사용' ON, AI 챗봇 전환 승인 여부를 확인하세요."
            )
            return jsonify(_process_payload_sync_budget(payload))

        # (c) 콜백 사용: 백그라운드로 돌리고 auto 모드면 아주 짧게 기다려 본다
        box: Dict[str, Any] = {}

        def worker() -> None:
            try:
                box["result"] = _process_payload(payload)
            except Exception as exc:  # pragma: no cover
                logger.exception("백그라운드 처리 중 예외: %s", exc)
                box["result"] = build_kakao_response(ANALYZE_FAIL_TEXT)

        thread = threading.Thread(target=worker, name="kakao-worker", daemon=True)
        thread.start()

        if CALLBACK_MODE == "auto":
            thread.join(max(0.3, CALLBACK_ACK_DELAY))
            if "result" in box:
                logger.info("제한시간 %.1fs 내 결과 확보 → 동기 응답", CALLBACK_ACK_DELAY)
                return jsonify(box["result"])
            logger.info("제한시간 초과 → 지연응답 반환 후 콜백 전송")
        else:
            logger.info("always 모드 → 지연응답 반환 후 콜백 전송")

        def deliver() -> None:
            thread.join(max(1.0, CALLBACK_DEADLINE))
            body = box.get("result") or build_kakao_response(ANALYZE_FAIL_TEXT)
            send_callback(callback_url, body, label="결과")

        threading.Thread(target=deliver, name="kakao-callback", daemon=True).start()
        return jsonify(build_callback_ack())

    except Exception as exc:  # 어떤 예외가 나도 카카오 규격으로 응답
        logger.exception("webhook 처리 중 예외: %s", exc)
        return jsonify(
            build_kakao_response("일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")
        )


# ---------------------------------------------------------------------------
# 10. 디버그 엔드포인트 (DEBUG_TOKEN 필요)
# ---------------------------------------------------------------------------
def _debug_authorized() -> bool:
    token = request.headers.get("X-Debug-Token") or request.args.get("token", "")
    return bool(DEBUG_TOKEN) and token == DEBUG_TOKEN


@app.route("/debug/extract", methods=["POST"])
def debug_extract() -> Any:
    """
    실제 카톡 페이로드를 그대로 붙여 넣어 '어느 단계에서 몇 개 추출·다운로드되는지' 본다.
    curl -X POST https://<앱>/debug/extract -H "X-Debug-Token: $DEBUG_TOKEN" -d @payload.json
    """
    if not DEBUG_TOKEN:
        return jsonify({"error": "DEBUG_TOKEN 환경변수가 설정되지 않았습니다."}), 403
    if not _debug_authorized():
        return jsonify({"error": "unauthorized"}), 401

    payload = _read_payload()
    urls, trace = extract_image_url_traced(payload)
    started = time.time()
    snapshots = snapshot_images(urls)
    return jsonify(
        {
            "request_id": str(uuid.uuid4()),
            "intent": _intent_label(payload),
            "extracted_count": len(urls),
            "extracted_urls_masked": [_mask_url(u) for u in urls],
            "downloaded_count": len(snapshots),
            "downloaded": [
                {
                    "url_masked": _mask_url(s.url),
                    "bytes": len(s.data),
                    "mime": s.mime,
                    "source": s.source,          # "kakao" | "storage"
                    "storage_key": s.key,
                    "storage_url": s.storage_url,
                }
                for s in snapshots
            ],
            "storage": (store.info() if store else {"backend": "none", "enabled": False}),
            "expire_epoch": extract_expire_epoch(payload),
            "download_seconds": round(time.time() - started, 3),
            "trace": trace,
            "top_level_keys": sorted(payload.keys()),
            "callback_url_present": bool(
                _as_dict(payload.get("userRequest")).get("callbackUrl")
            ),
            "question": _pick_question(payload),
            "raw_payload": payload,
        }
    )


@app.route("/debug/callback", methods=["POST"])
def debug_callback() -> Any:
    """
    callbackUrl 과 토큰이 실제로 살아 있는지 1건 테스트 전송한다.
    ★ 주의: callbackUrl 은 1건만 허용되므로, 이 호출이 실제 응답 1건을 소모한다.
      (직전 payload 에서 복사한 callbackUrl 은 수명이 5분이므로 즉시 테스트할 것)

    curl -X POST https://<앱>/debug/callback -H "X-Debug-Token: $DEBUG_TOKEN" \
         -d '{"callbackUrl":"https://bot-api.kakao.com/...","text":"테스트"}'
    """
    if not DEBUG_TOKEN:
        return jsonify({"error": "DEBUG_TOKEN 환경변수가 설정되지 않았습니다."}), 403
    if not _debug_authorized():
        return jsonify({"error": "unauthorized"}), 401

    body = _read_payload()
    callback_url = str(body.get("callbackUrl") or "").strip()
    text = str(body.get("text") or "콜백 연결 테스트입니다.").strip()
    if not callback_url:
        return jsonify({"error": "callbackUrl 이 필요합니다."}), 400

    ok = send_callback(callback_url, build_kakao_response(text), label="테스트")
    return jsonify({"sent": ok, "callback_url_host": urlsplit(callback_url).netloc})


@app.route("/debug/storage", methods=["POST", "GET"])
def debug_storage() -> Any:
    """
    스토리지 설정이 실제로 동작하는지 확인한다(1x1 PNG 업로드 → 재조회 → 바이트 비교).

    curl -X POST https://<앱>/debug/storage -H "X-Debug-Token: $DEBUG_TOKEN"
    """
    if not DEBUG_TOKEN:
        return jsonify({"error": "DEBUG_TOKEN 환경변수가 설정되지 않았습니다."}), 403
    if not _debug_authorized():
        return jsonify({"error": "unauthorized"}), 401

    if store is None or not store.enabled():
        return jsonify(
            {
                "ok": False,
                "reason": "스토리지가 비활성입니다. STORAGE_BACKEND 와 자격증명을 확인하세요.",
                "storage": (store.info() if store else {"backend": "none", "enabled": False}),
            }
        )

    probe = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB"
        "hwF/AAAAAElFTkSuQmCC"
    )
    key = "%s/_healthcheck/probe.png" % STORAGE_PREFIX
    started = time.time()
    uploaded_url = store.put(key, probe, "image/png")
    put_seconds = round(time.time() - started, 3)

    started = time.time()
    fetched = store.get(key)
    get_seconds = round(time.time() - started, 3)

    return jsonify(
        {
            "ok": bool(uploaded_url) and bool(fetched) and fetched[0] == probe,
            "uploaded_url": uploaded_url,
            "roundtrip_match": bool(fetched) and fetched[0] == probe,
            "bytes": len(fetched[0]) if fetched else 0,
            "put_seconds": put_seconds,
            "get_seconds": get_seconds,
            "storage": store.info(),
        }
    )


# ---------------------------------------------------------------------------
# 11. Render 슬립 방지(선택)
# ---------------------------------------------------------------------------
def _start_keepalive() -> None:
    """
    Render 무료 플랜은 15분 유휴 시 슬립한다. 슬립 후 첫 요청이 20~50초 걸려
    카카오 5초 제한을 무조건 넘긴다. KEEPALIVE_URL 을 자기 주소로 지정하면
    10분마다 스스로를 깨워 둔다. (유료 플랜이면 필요 없음)
    """
    if not KEEPALIVE_URL:
        return

    def loop() -> None:
        while True:
            time.sleep(max(60.0, KEEPALIVE_INTERVAL))
            try:
                response = requests.get(KEEPALIVE_URL, timeout=20)
                logger.info("keep-alive %s → HTTP %s", KEEPALIVE_URL, response.status_code)
            except Exception as exc:
                logger.warning("keep-alive 실패: %s", exc)

    threading.Thread(target=loop, name="keepalive", daemon=True).start()
    logger.info("keep-alive 스레드 시작 (%.0f초 간격, %s)", KEEPALIVE_INTERVAL, KEEPALIVE_URL)


if __name__ == "__main__":
    port = _env_int("PORT", 5000)
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY 가 비어 있습니다. .env 또는 환경변수를 설정하세요.")
    logger.info(
        "기동 v%s: port=%s, models=%s, callback=%s, sync_budget=%.1fs, pillow=%s, storage=%s",
        VERSION, port, _model_candidates(), CALLBACK_MODE, SYNC_BUDGET, _HAS_PIL,
        (store.info() if store else "none"),
    )
    if not _env_bool("FLASK_DEBUG", False):
        _start_keepalive()
    app.run(
        host="0.0.0.0", port=port,
        debug=_env_bool("FLASK_DEBUG", False), threaded=True,
    )
else:
    # gunicorn 으로 뜰 때도 keep-alive 시작(프리로드 미사용 전제)
    _start_keepalive()

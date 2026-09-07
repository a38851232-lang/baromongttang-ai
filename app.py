import os
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.get("/")
def home():
    return "바로몽땅찾아 서버 정상 작동 중", 200

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.post("/photo")
def photo():
    data = request.get_json(silent=True) or {}

    return jsonify({
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": "사진을 잘 받았습니다. 바로몽땅찾아 AI 분석 연결을 준비 중입니다."
                    }
                }
            ]
        }
    })

@app.post("/kakao/photo")
def kakao_photo():
    return photo()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

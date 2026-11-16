"""Flask-backend застосунку «AI-перекладач емоцій».

Маршрути:
  GET  /             — веб-інтерфейс
  POST /api/analyze  — аналіз емоції в тексті (JSON)
  GET  /api/model    — інформація про модель
  GET  /api/health   — перевірка роботи сервера
"""
import random

from flask import Flask, jsonify, render_template, request

from emotion_model import MAX_TEXT_LENGTH, MIN_TEXT_LENGTH, create_classifier
from emotions import EMOTIONS

EXAMPLES = [
    "Я так щасливий, що склав іспит на відмінно!",
    "Мені дуже сумно, всі друзі роз'їхалися.",
    "Це обурливо! Вони знову обманули клієнтів!",
    "Мені страшно йти додому самій у темряві.",
    "Ого, оце несподіванка, я такого не очікував!",
    "Завтра лекція починається о дев'ятій годині.",
]


def error(message: str, code: str, status: int):
    return jsonify({"error": message, "code": code}), status


def create_app(classifier=None) -> Flask:
    app = Flask(__name__)
    app.json.ensure_ascii = False                 # українські літери в JSON без \uXXXX
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # відсікаємо надто великі запити
    clf = classifier or create_classifier()

    @app.get("/")
    def index():
        return render_template("index.html", max_len=MAX_TEXT_LENGTH, examples=EXAMPLES)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/model")
    def model_info():
        return jsonify({
            "backend": clf.name,
            "classes": [{"key": k, "label": v["label"], "emoji": v["emoji"]}
                        for k, v in EMOTIONS.items()],
            "train_size": clf.train_size,
            "max_text_length": MAX_TEXT_LENGTH,
        })

    @app.post("/api/analyze")
    def analyze():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            return error("Очікується JSON виду {\"text\": \"...\"}.", "bad_request", 400)

        text = payload["text"].strip()
        if not text:
            return error("Введіть текст для аналізу.", "empty_text", 400)
        if len(text) < MIN_TEXT_LENGTH:
            return error(f"Текст занадто короткий (мінімум {MIN_TEXT_LENGTH} символи).",
                         "too_short", 400)
        if len(text) > MAX_TEXT_LENGTH:
            return error(f"Текст занадто довгий (максимум {MAX_TEXT_LENGTH} символів).",
                         "too_long", 400)

        pred = clf.predict(text)
        meta = EMOTIONS[pred.emotion]
        ranked = sorted(pred.probabilities.items(), key=lambda kv: kv[1], reverse=True)

        return jsonify({
            "emotion": pred.emotion,
            "label": meta["label"],
            "emoji": meta["emoji"],
            "color": meta["color"],
            "confidence": round(pred.confidence, 4),
            "low_confidence": pred.low_confidence,
            "message": meta["message"],
            "advice": random.choice(meta["advice"]),
            "probabilities": [
                {"emotion": k, "label": EMOTIONS[k]["label"],
                 "emoji": EMOTIONS[k]["emoji"], "probability": round(p, 4)}
                for k, p in ranked
            ],
        })

    @app.errorhandler(404)
    def not_found(_):
        if request.path.startswith("/api/"):
            return error("Ресурс не знайдено.", "not_found", 404)
        return render_template("index.html", max_len=MAX_TEXT_LENGTH, examples=EXAMPLES), 404

    @app.errorhandler(405)
    def method_not_allowed(_):
        return error("Метод не підтримується для цього маршруту.", "method_not_allowed", 405)

    @app.errorhandler(413)
    def too_large(_):
        return error("Запит занадто великий.", "payload_too_large", 413)

    @app.errorhandler(500)
    def server_error(_):
        return error("Внутрішня помилка сервера.", "server_error", 500)

    return app


app = create_app()

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)

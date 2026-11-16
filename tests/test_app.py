"""Автотести backend: API, валідація, ML-модель, обгортка HuggingFace (зі «заглушкою»).

Запуск із кореня проєкту:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import create_app                                    # noqa: E402
from emotion_model import (HuggingFaceEmotionClassifier, MAX_TEXT_LENGTH,  # noqa: E402
                           SklearnEmotionClassifier, normalize)
from emotions import EMOTIONS                                 # noqa: E402

CLF = SklearnEmotionClassifier()          # навчаємо один раз для всіх тестів
APP = create_app(CLF)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = APP.test_client()

    def post(self, payload):
        return self.client.post("/api/analyze", json=payload)

    # ---- сторінки та службові маршрути ----
    def test_index_page(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Перекладач емоцій", r.get_data(as_text=True))

    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.get_json(), {"status": "ok"})

    def test_model_info(self):
        data = self.client.get("/api/model").get_json()
        self.assertEqual(len(data["classes"]), 6)
        self.assertEqual(data["train_size"], 330)
        self.assertIn("scikit-learn", data["backend"])

    # ---- основний сценарій ----
    def test_analyze_sadness(self):
        r = self.post({"text": "Мені так сумно, я сумую за домом і хочу плакати"})
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertEqual(d["emotion"], "sadness")
        self.assertEqual(d["label"], "Сум")
        self.assertIn(d["advice"], EMOTIONS["sadness"]["advice"])

    def test_analyze_joy_and_anger(self):
        self.assertEqual(self.post({"text": "Я такий щасливий, ми перемогли!"}).get_json()["emotion"], "joy")
        self.assertEqual(self.post({"text": "Це обурливо, я розлючений на їхнє хамство!"}).get_json()["emotion"], "anger")

    def test_probabilities_are_valid(self):
        d = self.post({"text": "Мені страшно йти вночі самому"}).get_json()
        probs = [p["probability"] for p in d["probabilities"]]
        self.assertEqual(len(probs), 6)
        self.assertAlmostEqual(sum(probs), 1.0, places=2)
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertEqual(d["probabilities"][0]["emotion"], d["emotion"])

    def test_cyrillic_not_escaped_in_json(self):
        raw = self.post({"text": "Мені сумно"}).get_data(as_text=True)
        self.assertNotIn("\\u04", raw)

    def test_low_confidence_flag_for_gibberish(self):
        d = self.post({"text": "asdf qwerty"}).get_json()
        self.assertTrue(d["low_confidence"])
        self.assertLess(d["confidence"], 0.45)

    # ---- валідація ----
    def test_empty_and_whitespace(self):
        for text in ["", "     "]:
            r = self.post({"text": text})
            self.assertEqual(r.status_code, 400)
            self.assertEqual(r.get_json()["code"], "empty_text")

    def test_too_short(self):
        r = self.post({"text": "ок"})
        self.assertEqual((r.status_code, r.get_json()["code"]), (400, "too_short"))

    def test_too_long(self):
        r = self.post({"text": "а" * (MAX_TEXT_LENGTH + 1)})
        self.assertEqual((r.status_code, r.get_json()["code"]), (400, "too_long"))

    def test_bad_payloads(self):
        for payload in [{}, {"text": 123}, {"text": None}, ["text"], "text"]:
            r = self.post(payload)
            self.assertEqual(r.status_code, 400, payload)
            self.assertEqual(r.get_json()["code"], "bad_request")

    def test_invalid_json_and_wrong_content_type(self):
        r = self.client.post("/api/analyze", data="не json", content_type="application/json")
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/analyze", data="text=hello")
        self.assertEqual(r.status_code, 400)

    def test_payload_too_large(self):
        r = self.client.post("/api/analyze", data="x" * 20000, content_type="application/json")
        self.assertEqual(r.status_code, 413)

    def test_unknown_api_route_and_method(self):
        r = self.client.get("/api/nope")
        self.assertEqual((r.status_code, r.get_json()["code"]), (404, "not_found"))
        r = self.client.get("/api/analyze")
        self.assertEqual((r.status_code, r.get_json()["code"]), (405, "method_not_allowed"))


class ModelTests(unittest.TestCase):
    def test_low_confidence_threshold(self):
        from emotion_model import LOW_CONFIDENCE, Prediction
        self.assertTrue(Prediction("joy", LOW_CONFIDENCE - 0.01).low_confidence)
        self.assertFalse(Prediction("joy", LOW_CONFIDENCE + 0.01).low_confidence)

    def test_normalize_apostrophes_and_case(self):
        self.assertEqual(normalize("  Дев’ятий   ПОВЕРХ "), "дев'ятий поверх")

    def test_inflected_forms_are_recognised(self):
        # символьні n-грами стійкі до словозміни
        self.assertEqual(CLF.predict("Я сумую, мені дуже сумно").emotion, "sadness")

    def test_neutral_statement(self):
        self.assertEqual(CLF.predict("Лекція починається о десятій годині").emotion, "neutral")


class HuggingFaceWrapperTests(unittest.TestCase):
    """Перевіряємо лише логіку обгортки (без завантаження реальної моделі)."""

    def test_maps_labels_back_to_keys(self):
        calls = {}

        def fake_pipe(text, candidate_labels, hypothesis_template):
            calls["labels"] = candidate_labels
            calls["template"] = hypothesis_template
            return {"labels": ["сум", "страх", "радість", "гнів", "здивування", "спокій"],
                    "scores": [0.6, 0.2, 0.1, 0.05, 0.03, 0.02]}

        hf = HuggingFaceEmotionClassifier(model_name="stub", pipe=fake_pipe)
        pred = hf.predict("тест")
        self.assertEqual(pred.emotion, "sadness")
        self.assertAlmostEqual(pred.confidence, 0.6)
        self.assertEqual(set(pred.probabilities), set(EMOTIONS))
        self.assertEqual(set(calls["labels"]), {m["label"].lower() for m in EMOTIONS.values()})

    def test_api_works_with_hf_wrapper(self):
        def fake_pipe(text, candidate_labels, hypothesis_template):
            return {"labels": ["радість"] + [l for l in candidate_labels if l != "радість"],
                    "scores": [0.5, 0.2, 0.1, 0.1, 0.05, 0.05]}
        client = create_app(HuggingFaceEmotionClassifier("stub", pipe=fake_pipe)).test_client()
        d = client.post("/api/analyze", json={"text": "привіт, як справи"}).get_json()
        self.assertEqual(d["emotion"], "joy")


if __name__ == "__main__":
    unittest.main(verbosity=2)

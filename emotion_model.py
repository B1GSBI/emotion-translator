"""ML-ядро застосунку: класифікація емоцій у тексті українською мовою.

За замовчуванням використовується легка модель scikit-learn
(TF-IDF символьних n-грам + логістична регресія), яка
навчається при старті сервера на невеликому датасеті data/emotions_uk.csv.
Опційно можна ввімкнути трансформерну модель HuggingFace
(змінна середовища EMOTION_BACKEND=hf).
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from emotions import EMOTIONS

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "emotions_uk.csv"
HOLDOUT_PATH = BASE_DIR / "data" / "holdout_uk.csv"

MAX_TEXT_LENGTH = 1000
MIN_TEXT_LENGTH = 3
# Нижче цього порогу модель вважається невпевненою. Значення підібрано за out-of-fold
# ймовірностями крос-валідації: точність для впевнених відповідей ≈ 90 %, для невпевнених ≈ 56 %.
LOW_CONFIDENCE = 0.45


@dataclass
class Prediction:
    emotion: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def low_confidence(self) -> bool:
        return self.confidence < LOW_CONFIDENCE


def normalize(text: str) -> str:
    """Уніфікація тексту: регістр, апострофи, зайві пробіли."""
    text = text.lower().replace("’", "'").replace("ʼ", "'").replace("`", "'")
    return re.sub(r"\s+", " ", text).strip()


def load_dataset(path: Path = DATA_PATH) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def build_pipeline() -> Pipeline:
    """TF-IDF символьних n-грам (стійкі до відмінків і словоформ) + логістична регресія.

    Гіперпараметри підібрано повторною крос-валідацією на навчальному датасеті;
    окремий hold-out набір у підборі не використовувався.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
            preprocessor=normalize)),
        ("clf", LogisticRegression(C=100, max_iter=3000)),
    ])


class SklearnEmotionClassifier:
    """Класифікатор на основі scikit-learn."""

    name = "scikit-learn: TF-IDF (символьні n-грами) + LogisticRegression"

    def __init__(self, data_path: Path = DATA_PATH):
        texts, labels = load_dataset(data_path)
        self.train_size = len(texts)
        self.pipeline = build_pipeline().fit(texts, labels)
        self.classes = [str(c) for c in self.pipeline.classes_]

    def predict(self, text: str) -> Prediction:
        proba = self.pipeline.predict_proba([text])[0]
        probabilities = {c: float(p) for c, p in zip(self.classes, proba)}
        best = max(probabilities, key=probabilities.get)
        return Prediction(best, probabilities[best], probabilities)


class HuggingFaceEmotionClassifier:
    """Опційний режим: zero-shot класифікація багатомовною трансформерною моделлю.

    Потребує пакетів transformers і torch та завантаження ваг моделі
    (сотні МБ), тому на безкоштовному хостингу з 512 МБ RAM не рекомендується.
    """

    DEFAULT_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    TEMPLATE = "Цей текст виражає {}."

    def __init__(self, model_name: str | None = None, pipe=None):
        self.model_name = model_name or os.getenv("HF_MODEL", self.DEFAULT_MODEL)
        if pipe is None:
            from transformers import pipeline  # відкладений імпорт
            pipe = pipeline("zero-shot-classification", model=self.model_name)
        self._pipe = pipe
        self._by_label = {meta["label"].lower(): key for key, meta in EMOTIONS.items()}
        self.name = f"HuggingFace Transformers (zero-shot): {self.model_name}"
        self.train_size = 0
        self.classes = list(EMOTIONS)

    def predict(self, text: str) -> Prediction:
        result = self._pipe(
            text,
            candidate_labels=list(self._by_label),
            hypothesis_template=self.TEMPLATE,
        )
        probabilities = {self._by_label[l]: float(s)
                         for l, s in zip(result["labels"], result["scores"])}
        best = max(probabilities, key=probabilities.get)
        return Prediction(best, probabilities[best], probabilities)


def create_classifier():
    """Фабрика: вибір бекенду за змінною середовища EMOTION_BACKEND."""
    backend = os.getenv("EMOTION_BACKEND", "sklearn").lower()
    if backend == "hf":
        return HuggingFaceEmotionClassifier()
    return SklearnEmotionClassifier()

"""Оцінка якості ML-моделі: 5-fold крос-валідація та окремий hold-out набір.

Запуск:  python evaluate.py
Результати також зберігаються у docs/metrics.json.
"""
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from emotion_model import DATA_PATH, HOLDOUT_PATH, build_pipeline, load_dataset


def main():
    X, y = load_dataset(DATA_PATH)
    Xh, yh = load_dataset(HOLDOUT_PATH)
    labels = sorted(set(y))

    # 1) 5-fold стратифікована крос-валідація на навчальному датасеті
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_cv = cross_val_predict(build_pipeline(), X, y, cv=cv)
    cv_acc = accuracy_score(y, y_cv)
    cv_f1 = f1_score(y, y_cv, average="macro")

    # 2) навчання на всьому датасеті й перевірка на окремих реченнях (hold-out)
    model = build_pipeline().fit(X, y)
    y_h = model.predict(Xh)
    h_acc = accuracy_score(yh, y_h)
    h_f1 = f1_score(yh, y_h, average="macro")

    print(f"Навчальних речень: {len(X)}, hold-out: {len(Xh)}\n")
    print(f"[CV 5-fold]  accuracy = {cv_acc:.3f}, macro-F1 = {cv_f1:.3f}")
    print(f"[Hold-out]   accuracy = {h_acc:.3f}, macro-F1 = {h_f1:.3f}\n")
    print("Звіт (hold-out):")
    print(classification_report(yh, y_h, labels=labels, digits=2, zero_division=0))
    cm = confusion_matrix(yh, y_h, labels=labels)
    print("Матриця помилок (hold-out), рядки — істинні, стовпці — передбачені:")
    print("labels:", labels)
    print(cm)

    out = {
        "train_size": len(X), "holdout_size": len(Xh), "labels": labels,
        "cv_accuracy": round(cv_acc, 4), "cv_macro_f1": round(cv_f1, 4),
        "holdout_accuracy": round(h_acc, 4), "holdout_macro_f1": round(h_f1, 4),
        "holdout_report": classification_report(yh, y_h, labels=labels, output_dict=True, zero_division=0),
        "holdout_confusion_matrix": cm.tolist(),
    }
    Path("docs").mkdir(exist_ok=True)
    Path("docs/metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

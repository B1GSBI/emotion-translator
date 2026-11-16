# Перекладач емоцій

Навчальний веб-застосунок (практична робота № 9): користувач вводить текст українською, а
модель машинного навчання визначає емоцію (радість, сум, гнів, страх, здивування, спокій) і
показує емодзі, пораду та розподіл імовірностей.

**Стек:** Python 3.10+, Flask (backend + REST API), scikit-learn (ML-модель), HTML/CSS/JS (інтерфейс), gunicorn (продакшн-сервер).

## Структура

```
app.py              Flask-backend: маршрути, валідація, JSON API
emotion_model.py    ML-ядро: TF-IDF (символьні n-грами) + LogisticRegression; опційний HuggingFace-режим
emotions.py         Емоції: назви, емодзі, кольори, поради
evaluate.py         Оцінка якості моделі (крос-валідація + hold-out) -> docs/metrics.json
data/               emotions_uk.csv (330 речень, навчання), holdout_uk.csv (36 речень, перевірка)
templates/, static/ Веб-інтерфейс
tests/test_app.py   Автотести (unittest)
render.yaml, Procfile   Конфігурація публікації
```

## Запуск локально

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                      # http://localhost:5000
```

Модель навчається при старті сервера (~1-2 с), окремого файлу моделі не потрібно.

## Тести та оцінка моделі

```bash
python -m unittest discover -s tests -v   # 21 тест: API, валідація, модель
python evaluate.py                        # метрики якості
```

## API

| Метод | Шлях | Опис |
|-------|------|------|
| GET | `/` | Веб-інтерфейс |
| POST | `/api/analyze` | Аналіз тексту. Тіло: `{"text": "..."}` |
| GET | `/api/model` | Інформація про модель і класи |
| GET | `/api/health` | Перевірка роботи сервера |

```bash
curl -X POST http://localhost:5000/api/analyze \
     -H "Content-Type: application/json" \
     -d '{"text": "Мені дуже сумно, всі друзі роз'"'"'їхалися"}'
```

Помилки повертаються у вигляді `{"error": "...", "code": "..."}` зі статусом 400/404/405/413.

## Публікація на Render (безкоштовний тариф)

1. Завантажте проєкт у репозиторій GitHub (`git init`, `git add .`, `git commit`, `git push`).
2. Зареєструйтесь на https://render.com і підключіть GitHub.
3. **New → Web Service** → оберіть репозиторій. Параметри:
   - Runtime: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --workers 1 --threads 4 --timeout 120`
   - Instance Type: `Free`
   - Environment: `PYTHON_VERSION` = `3.12.3`
   (або **New → Blueprint** — Render прочитає `render.yaml` автоматично)
4. Натисніть **Create Web Service**. Через 2-3 хвилини застосунок буде доступний за адресою
   `https://<назва>.onrender.com`.

Особливості безкоштовного тарифу: сервіс «засинає» після 15 хвилин без запитів, перший запит
після простою триває близько хвилини; файлова система тимчасова (застосунок нічого не записує на диск).

## Опційно: трансформерна модель HuggingFace

Режим `EMOTION_BACKEND=hf` вмикає zero-shot класифікацію багатомовною моделлю
(`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, змінюється через `HF_MODEL`).
Потрібні додаткові пакети: `pip install transformers torch`; ваги моделі (сотні МБ)
завантажуються при першому запуску. Для безкоштовного тарифу Render (512 МБ RAM) цей режим не підходить.

## Обмеження

- Модель навчена на невеликому наборі речень (330), тому може помилятися, особливо на іронії,
  складних і неоднозначних текстах. Для таких випадків API повертає `low_confidence: true`.
- Застосунок навчальний і не є медичною чи психологічною діагностикою.
- Тексти користувачів не зберігаються.

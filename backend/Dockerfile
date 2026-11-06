FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models \
    DB_PATH=/app/data/predictions.db

WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + serialized models (inference logic stays immutable)
COPY cardio_ml/ /app/cardio_ml/
COPY app.py db.py /app/
COPY models/ /app/models/

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

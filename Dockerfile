FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY . .

RUN mkdir -p data/raw data/processed data/eval \
    checkpoints rag api/query_log.jsonl

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD wget --no-verbose --tries=1 \
  --spider http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]

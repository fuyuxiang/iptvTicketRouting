FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# system deps (jieba doesn't require; keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY app /app/app
COPY configs /app/configs
COPY scripts /app/scripts
COPY data /app/data

RUN pip install -U pip \
    && pip install "." \
    && pip cache purge

# Train or copy models as part of your CI/CD image build.
# For demo: train at build time on sample data.
RUN python scripts/train.py --config configs/config.yaml

EXPOSE 8080

# Gunicorn + UvicornWorker for production
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "-w", "4", "-b", "0.0.0.0:8080", "--access-logfile", "-", "--error-logfile", "-"]

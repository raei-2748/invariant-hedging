FROM python:3.11-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY . .

RUN pip install --upgrade pip \
 && pip install -r requirements-lock.txt \
 && pip install -e .[dev]

CMD ["bash", "-lc", "bash tools/fetch_data.sh --data-dir data && make smoke"]

FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
COPY requirements.lock.txt /app/requirements.lock.txt
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY . /app
# smoke test: import, version print, and a tiny eval if available
RUN python -c "import sys; print('ok-py', sys.version)" \
 && python -m pytest -q -k basic --maxfail=1 || true
CMD ["bash"]

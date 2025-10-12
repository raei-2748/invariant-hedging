# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG MICROMAMBA_VERSION=1.5.8-0

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        git \
        tini \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L "https://micromamba.snakepit.net/api/micromamba/linux-64/${MICROMAMBA_VERSION}/micromamba.tar.bz2" \
    | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba

COPY environment.yml requirements.txt /tmp/env/

RUN micromamba create -y -f /tmp/env/environment.yml && \
    micromamba clean -a -y

WORKDIR /workspace/invariant-hedging

COPY . /workspace/invariant-hedging

SHELL ["/usr/local/bin/micromamba", "run", "-n", "invariant-hedging", "/bin/bash", "-c"]

RUN python -m pip install --no-deps -r requirements.txt && \
    python -m pip check

ENTRYPOINT ["/usr/local/bin/micromamba", "run", "-n", "invariant-hedging"]
CMD ["bash"]

# syntax=docker/dockerfile:1.7
ARG ROCM_BASE_IMAGE=rocm/pytorch:latest

############################
# CPU builder
############################
FROM python:3.11-slim AS builder-cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/opt/venv

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:${PATH}"

COPY requirements.base.txt ./
COPY requirements.train.txt ./
COPY requirements.cpu.txt ./

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.base.txt && \
    pip install -r requirements.train.txt && \
    pip install -r requirements.cpu.txt

############################
# NVIDIA builder
############################
FROM python:3.11-slim AS builder-nvidia

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/opt/venv

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:${PATH}"

COPY requirements.base.txt ./
COPY requirements.train.txt ./
COPY requirements.nvidia.txt ./

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.base.txt && \
    pip install -r requirements.train.txt && \
    pip install -r requirements.nvidia.txt

############################
# API builder
############################
FROM python:3.11-slim AS builder-api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VENV_PATH=/opt/venv

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "${VENV_PATH}"
ENV PATH="${VENV_PATH}/bin:${PATH}"

COPY requirements.base.txt ./
COPY requirements.api.txt ./

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.base.txt && \
    pip install -r requirements.api.txt

############################
# CPU runtime
############################
FROM python:3.11-slim AS runtime-cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder-cpu /opt/venv /opt/venv
RUN mkdir -p /app/data /app/outputs /app/src /app/scripts /app/services /app/configs /mlruns

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]

############################
# NVIDIA runtime
############################
FROM python:3.11-slim AS runtime-nvidia

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder-nvidia /opt/venv /opt/venv
RUN mkdir -p /app/data /app/outputs /app/src /app/scripts /app/services /app/configs /mlruns

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]

############################
# API runtime
############################
FROM python:3.11-slim AS runtime-api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_PATH=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app:/app/services/embedding-api

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder-api /opt/venv /opt/venv
RUN mkdir -p /app/data /app/outputs /app/src /app/services /app/configs

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]

############################
# AMD runtime
############################
FROM ${ROCM_BASE_IMAGE} AS runtime-amd

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.base.txt ./
COPY requirements.train.txt ./

RUN pip install -r requirements.base.txt && \
    pip install -r requirements.train.txt

RUN mkdir -p /app/data /app/outputs /app/src /app/scripts /app/services /app/configs /mlruns

CMD ["bash"]

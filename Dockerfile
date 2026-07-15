# syntax=docker/dockerfile:1
# BASE_IMAGE: "cpu" or "cuda" - selects which base image to use
ARG BASE_IMAGE=cpu

# GoC Root-A cert — injected before uv sync so TLS interception (ICM) does not break the build
ARG GOC_ROOT_A_URL="https://raw.githubusercontent.com/gccloudone-aurora-collab/goc-root-cert-mirror/main/certs/GoC-GdC-Root-A.crt"
ARG GOC_ROOT_A_FINGERPRINT="FE:E0:9E:77:43:BF:D4:3E:D7:D4:D3:ED:50:6C:C7:9D:2D:90:70:FF:A9:29:91:16:87:D4:27:33:70:BE:A3:06"

# ── CUDA builder ───────────────────────────────────────────────────────────────
# nvidia/cuda devel image is needed to compile flash-attn; it is NOT used at runtime.
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS builder-cuda
# For local dev:  docker build --build-arg BASE_IMAGE=cuda --build-arg CUDA_ARCH="8.6" -t ekh:local-8.6 .
# Version that must match the GPU you are running this service on --> https://developer.nvidia.com/cuda/gpus
ARG CUDA_ARCH
ARG JOBS_AND_THREADS=8
ARG BUILD_FLASH=FALSE

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git openssl python3.12 python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Inject GoC Root-A cert so uv sync can reach PyPI through the ICM proxy
ARG GOC_ROOT_A_URL
ARG GOC_ROOT_A_FINGERPRINT
RUN curl -fsSL --insecure "${GOC_ROOT_A_URL}" \
        -o /usr/local/share/ca-certificates/GoC-GdC-Root-A.crt \
    && test "$(openssl x509 -in /usr/local/share/ca-certificates/GoC-GdC-Root-A.crt -noout -sha256 -fingerprint | cut -d= -f2)" = "${GOC_ROOT_A_FINGERPRINT}" \
    && update-ca-certificates

# Install UV (build stage only - not carried to runtime)
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV UV_NO_DEV=1
ENV UV_PYTHON_PREFERENCE=only-system
ENV PATH="/root/.local/bin/:$PATH"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="$CUDA_HOME/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock ./

# Compile flash-attn from source (no binary wheel exists).
# All .o / .ptx / .cubin build artefacts and the UV cache stay in THIS stage only
# and are automatically discarded when the final runtime stage is assembled.
RUN export MAX_JOBS=${JOBS_AND_THREADS} && \
    export NVCC_THREADS=${JOBS_AND_THREADS} && \
    export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" && \
    export FLASH_ATTENTION_FORCE_BUILD="TRUE" && \
    export FLASH_ATTENTION_FORCE_CXX11_ABI="FALSE" && \
    export FLASH_ATTENTION_SKIP_CUDA_BUILD="FALSE" && \
    export FLASH_ATTN_CUDA_ARCHS="$(echo ${CUDA_ARCH} | tr -d '.')" && \
    FLASH_EXTRA="" && \
    if [ "${BUILD_FLASH}" = "TRUE" ]; then FLASH_EXTRA="--extra flash"; fi && \
    uv sync ${FLASH_EXTRA} --locked

# ── CPU builder ────────────────────────────────────────────────────────────────
FROM python:3.12-trixie AS builder-cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git openssl \
    && rm -rf /var/lib/apt/lists/*

# Inject GoC Root-A cert so uv sync can reach PyPI through the ICM proxy
ARG GOC_ROOT_A_URL
ARG GOC_ROOT_A_FINGERPRINT
RUN curl -fsSL --insecure "${GOC_ROOT_A_URL}" \
        -o /usr/local/share/ca-certificates/GoC-GdC-Root-A.crt \
    && test "$(openssl x509 -in /usr/local/share/ca-certificates/GoC-GdC-Root-A.crt -noout -sha256 -fingerprint | cut -d= -f2)" = "${GOC_ROOT_A_FINGERPRINT}" \
    && update-ca-certificates

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV UV_NO_DEV=1
ENV UV_PYTHON_PREFERENCE=only-system
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN uv sync --locked

# ── Alias: select the right builder under a fixed name ───────────────────────
# Docker does not support build args in COPY --from, but does support them in
# FROM.  Creating this alias stage gives COPY a stable literal name to target.
ARG BASE_IMAGE
FROM builder-${BASE_IMAGE} AS builder

# ── CUDA runtime ───────────────────────────────────────────────────────────────
# cudnn-runtime is ~4 GB lighter than cudnn-devel; no compiler/headers.
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 AS runtime-cuda
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
    && rm -rf /var/lib/apt/lists/*

# ── CPU runtime ────────────────────────────────────────────────────────────────
# python:slim contains only the interpreter + minimal OS; no compiler toolchain.
FROM python:3.12-slim-trixie AS runtime-cpu

# ── Final stage ────────────────────────────────────────────────────────────────
ARG BASE_IMAGE
FROM runtime-${BASE_IMAGE} AS final

WORKDIR /app

# Carry the GoC Root-A cert (and the full updated bundle) from the builder
# so the running application can make TLS calls through the ICM proxy.
COPY --from=builder /usr/local/share/ca-certificates/ /usr/local/share/ca-certificates/
COPY --from=builder /etc/ssl/certs/ /etc/ssl/certs/

# Copy only the pre-built virtual environment from the appropriate builder.
# Nothing else from the builder (UV cache, build tools, CUDA devel files,
# flash-attn .o/.ptx artefacts, apt lists) is carried over.
COPY --from=builder /app/.venv /app/.venv

COPY main.py ./
COPY provider ./provider
COPY repository ./repository
COPY router ./router
COPY services ./services

# Use the venv directly - no UV needed at runtime
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV=/app/.venv

CMD ["python", "-m", "fastapi", "run", "main.py"]
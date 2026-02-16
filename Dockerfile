# Build args must be declared before FROM to use in image selection
ARG ENABLE_CUDA=false
ARG CUDA_ARCH

# Select base image based on CUDA flag
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS base-cuda
FROM python:3.12-trixie AS base-cpu
FROM base-${ENABLE_CUDA:+cuda}${ENABLE_CUDA:-cpu} AS base

# For local dev:  docker build --build-arg ENABLE_CUDA=true --build-arg CUDA_ARCH="8.6" -t ekh:local-8.6 .
ARG ENABLE_CUDA
ARG CUDA_ARCH

# install uv (from https://docs.astral.sh/uv/guides/integration/docker/#installing-uv)
# The installer requires curl (and certificates) to download the release archive
# git is required for flash-attn to fetch CUTLASS submodule during build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates git

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV UV_NO_DEV=1
ENV PATH="/root/.local/bin/:$PATH"

# CUDA-specific environment variables (only set when CUDA is enabled)
# flash-attn build configuration (see: https://github.com/Dao-AILab/flash-attention)
# MAX_JOBS: Limit parallel ninja jobs to avoid OOM during compilation
# NVCC_THREADS: Limit nvcc threads per job
# TORCH_CUDA_ARCH_LIST: Target GPU architectures (used by PyTorch cpp_extension)
RUN if [ "$ENABLE_CUDA" = "true" ]; then \
      echo "CUDA_HOME=/usr/local/cuda" >> /etc/environment && \
      echo "MAX_JOBS=4" >> /etc/environment && \
      echo "NVCC_THREADS=2" >> /etc/environment && \
      echo "TORCH_CUDA_ARCH_LIST=${CUDA_ARCH}" >> /etc/environment && \
      echo "FLASH_ATTENTION_FORCE_BUILD=TRUE" >> /etc/environment && \
      echo "FLASH_ATTENTION_FORCE_CXX11_ABI=FALSE" >> /etc/environment && \
      echo "FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE" >> /etc/environment; \
    fi

ENV CUDA_HOME=${ENABLE_CUDA:+/usr/local/cuda}
ENV PATH="${CUDA_HOME:+$CUDA_HOME/bin:}$PATH"

WORKDIR /app

COPY main.py ./
COPY provider ./provider
COPY repository ./repository
COPY router ./router
COPY services ./services

COPY pyproject.toml uv.lock ./

# Build with or without flash-attn based on CUDA flag
# FLASH_ATTN_CUDA_ARCHS needs shell expansion, so compute it inline
RUN if [ "$ENABLE_CUDA" = "true" ]; then \
      export CUDA_HOME=/usr/local/cuda && \
      export MAX_JOBS=4 && \
      export NVCC_THREADS=2 && \
      export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" && \
      export FLASH_ATTENTION_FORCE_BUILD="TRUE" && \
      export FLASH_ATTENTION_FORCE_CXX11_ABI="FALSE" && \
      export FLASH_ATTENTION_SKIP_CUDA_BUILD="FALSE" && \
      export FLASH_ATTN_CUDA_ARCHS="$(echo ${CUDA_ARCH} | tr -d '.')" && \
      uv sync --extra flash --locked; \
    else \
      uv sync --locked; \
    fi

CMD ["uv", "run", "fastapi", "run", "main.py"]
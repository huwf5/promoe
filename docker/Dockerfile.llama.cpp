ARG UBUNTU_VERSION=22.04

# This needs to generally match the container host's environment.
ARG CUDA_VERSION=12.4.1

# Target the CUDA build image
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_DEV_CONTAINER} AS build

# Unless otherwise specified, we make a fat build.
# ARG CUDA_DOCKER_ARCH=89

RUN apt-get update && \
    apt-get install -y build-essential python3 python3-pip git libcurl4-openssl-dev libgomp1 zsh curl wget vim ccache

COPY requirements.txt   requirements.txt
COPY requirements       requirements

RUN pip install --upgrade pip setuptools wheel pandas \
    && pip install -r requirements.txt \
    && pip install flash_attn optimum

WORKDIR /code

# COPY . .

# Set nvcc architecture
ENV CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH}
# Enable CUDA
ENV GGML_CUDA=1
# Enable cURL
ENV LLAMA_CURL=1

# RUN make -j$(nproc)

# ENTRYPOINT ["/app/.devops/tools.sh"]

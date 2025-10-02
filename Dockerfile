# base stage
FROM ubuntu:22.04 AS base
USER root
SHELL ["/bin/bash", "-c"]
WORKDIR /tiny_rag

ARG NEED_MIRROR=1
ENV DEBIAN_FRONTEND=noninteractive

# ============================================================================ #
# setup apt & install packages
RUN --mount=type=cache,id=tiny_rag_apt,target=/var/cache/apt,sharing=locked \
    if [ "$NEED_MIRROR" == "1" ]; then \
        sed -i 's|http://ports.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list; \
        sed -i 's|http://archive.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list; \
    fi; \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache && \
    chmod 1777 /tmp && \
    apt update && \
    apt --no-install-recommends install -y ca-certificates && \
    apt install -y default-jdk && \
    apt install -y build-essential && \
    apt install -y nginx unzip curl wget git vim less

# setup vim
RUN echo "set number" >> /etc/vim/vimrc

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# ============================================================================ #
# builder stage
# install dependencies from requirements.txt
FROM base AS builder
USER root
SHELL ["/bin/bash", "-c"]
WORKDIR /tiny_rag

ARG NEED_MIRROR=1
ENV DEBIAN_FRONTEND=noninteractive

# ============================================================================ #
# install python 3.12 and packages using uv
COPY pyproject.toml .
RUN --mount=type=cache,id=tiny_rag_uv,target=/root/.cache/uv,sharing=locked \
    if [ "$NEED_MIRROR" == "1" ]; then \
        export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple; \
    fi; \
    uv venv .venv --python 3.12 && \
    uv sync

# ============================================================================ #
# copy project files
COPY src/common common
COPY src/rag rag
COPY src/parse parse
COPY config.yaml .

COPY docker/entrypoint.sh .
RUN chmod +x ./entrypoint*.sh

# ============================================================================ #
# set up container entrypoint
ENTRYPOINT ["./entrypoint.sh"]
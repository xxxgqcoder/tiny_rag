# base stage
FROM ubuntu:22.04 AS base
USER root
SHELL ["/bin/bash", "-c"]

ARG NEED_MIRROR=0
WORKDIR /tiny_rag
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
    apt update && \
    apt install -y default-jdk && \
    apt install -y build-essential && \
    apt install -y python3-pip pipx nginx unzip curl wget git vim less

# setup vim
RUN echo "set number" >> /etc/vim/vimrc


# ============================================================================ #
# builder stage
# install dependencies from requirements.txt
FROM base AS builder
USER root
WORKDIR /tiny_rag

COPY requirements.txt .
RUN --mount=type=cache,id=tiny_rag_pip,target=/root/.cache/uv,sharing=locked \
    if [ "$NEED_MIRROR" == "1" ]; then \
        pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
        pip3 config set global.trusted-host mirrors.aliyun.com; \
    fi; \
    pip3 install -r requirements.txt


# ============================================================================ #
# copy project files
COPY assets assets
COPY parser parser
COPY config.py .
COPY utils.py .

COPY start_jupyter.sh .
RUN chmod +x start_jupyter.sh

COPY docker/entrypoint.sh .
RUN chmod +x ./entrypoint*.sh


# ============================================================================ #
# set up entrypoint
# NOTE: for debug only
ENTRYPOINT ["./entrypoint.sh"]


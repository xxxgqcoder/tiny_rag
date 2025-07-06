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
    apt update && \
    apt install -y default-jdk && \
    apt install -y build-essential && \
    apt install -y python3-pip pipx nginx unzip curl wget git vim less

# setup vim
RUN echo "set number" >> /etc/vim/vimrc

# python cmd
RUN ln -s /usr/bin/python3 /usr/bin/python


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
# install python packages
COPY requirements.txt .
RUN --mount=type=cache,id=tiny_rag_pip,target=/root/.cache/pip,sharing=locked \
    if [ "$NEED_MIRROR" == "1" ]; then \
        pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
        pip3 config set global.trusted-host mirrors.aliyun.com; \
    fi; \
    pip3 install -r requirements.txt

# ============================================================================ #
# copy project files
COPY assets assets
COPY parse parse
COPY rag rag
COPY config.py .
COPY utils.py .
COPY start_server.py .
COPY chat.py .

COPY notebooks .

COPY notebooks/start_jupyter.sh .
RUN chmod +x start_jupyter.sh

COPY entrypoint.sh .
RUN chmod +x ./entrypoint*.sh


# ============================================================================ #
# set up container entrypoint
ENTRYPOINT ["./entrypoint.sh"]

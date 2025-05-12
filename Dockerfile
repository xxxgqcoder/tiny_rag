# base stage
FROM ubuntu:22.04 AS base
USER root
SHELL ["/bin/bash", "-c"]

ARG NEED_MIRROR=0

WORKDIR /my_rag

ENV DEBIAN_FRONTEND=noninteractive


# setup apt & install packages
# python package and implicit dependencies:
#   opencv-python: libglib2.0-0 libglx-mesa0 libgl1
#   aspose-slides: pkg-config libicu-dev libgdiplus libssl1.1_1.1.1f-1ubuntu2_amd64.deb
#   python-pptx: default-jdk tika-server-standard-3.0.0.jar
# building C extensions: libpython3-dev libgtk-4-1 libnss3 xdg-utils libgbm-dev
RUN --mount=type=cache,id=my_rag_apt,target=/var/cache/apt,sharing=locked \
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
    apt install -y libglib2.0-0 libglx-mesa0 libgl1 && \
    apt install -y pkg-config libicu-dev libgdiplus && \
    apt install -y default-jdk && \
    apt install -y libpython3-dev libgtk-4-1 libnss3 xdg-utils libgbm-dev && \
    apt install -y libjemalloc-dev && \
    apt install -y python3-pip pipx nginx unzip curl wget git vim less


# setup vim
RUN echo "set number" >> /etc/vim/vimrc


# install uv
RUN if [ "$NEED_MIRROR" == "1" ]; then \
        pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
        pip3 config set global.trusted-host mirrors.aliyun.com; \
        mkdir -p /etc/uv && \
        echo "[[index]]" > /etc/uv/uv.toml && \
        echo 'url = "https://mirrors.aliyun.com/pypi/simple"' >> /etc/uv/uv.toml && \
        echo "default = true" >> /etc/uv/uv.toml; \
    fi; \
    pipx install uv


# builder stage
FROM base AS builder
USER root

COPY deepdoc /my_rag/deepdoc

WORKDIR /my_rag

COPY docker/entrypoint.sh .
RUN chmod +x ./entrypoint*.sh

# NOTE(guoqing): for debug only
# ENTRYPOINT ["./entrypoint.sh"]
ENTRYPOINT ["tail", "-f", "/dev/null"]


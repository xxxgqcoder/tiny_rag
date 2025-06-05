# base stage
FROM tiny_rag:dep AS base
USER root
SHELL ["/bin/bash", "-c"]

ARG NEED_MIRROR=0
WORKDIR /tiny_rag
ENV DEBIAN_FRONTEND=noninteractive

# ============================================================================ #
# builder stage
# install dependencies from requirements.txt
FROM base AS builder
USER root
WORKDIR /tiny_rag

COPY requirements.txt .
RUN --mount=type=cache,id=tiny_rag_pip,target=/root/.cache/pip,sharing=locked \
    if [ "$NEED_MIRROR" == "1" ]; then \
        pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
        pip3 config set global.trusted-host mirrors.aliyun.com; \
    fi; \
    pip3 install -r requirements.txt


# ============================================================================ #
# additional pip package
RUN --mount=type=cache,id=tiny_rag_pip_additional,target=/root/.cache/pip,sharing=locked \
    pip install ollama==0.4.9 prompt_toolkit==3.0.51 Flask==3.0.3 


# ============================================================================ #
# copy project files
COPY assets assets
COPY parse parse
COPY rag rag
COPY config.py .
COPY utils.py .
COPY start_server.py .
COPY chat.py .

COPY notebooks/start_jupyter.sh .
RUN chmod +x start_jupyter.sh

COPY entrypoint.sh .
RUN chmod +x ./entrypoint*.sh


# ============================================================================ #
# set up container entrypoint
ENTRYPOINT ["./entrypoint.sh"]

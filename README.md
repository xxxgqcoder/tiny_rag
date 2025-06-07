# Tiny RAG
Tiny RAG is a simplified LLM base RAG, which acts as personal knowledge assistance
and focuses only on core logic of an RAG system.

Key features 
1. Local directory base kowledge base management. All knowledge base files are stored in local file folder. Tiny RAG monitors local file directory. Once file created / deleted, Tiny RAG will automatically digest and delete corresponding content.
2. Advanced PDF parser library used. Tiny RAG adopts [`MinerU`](https://github.com/opendatalab/MinerU) for pdf parsing.
3. User [`Milvus`](https://milvus.io/) as vector db for fast and hybird (dense vector and key-word matching combined) content retrival.
4. Leverage local [`ollama`](https://ollama.com/) LLM for conversation.
5. Deploy via [`docker`](https://www.docker.com/).



# Hot to Deploy

## Install Docker and Ollama
To deploy this project, you need to install below two softwares.
- [`docker`](https://www.docker.com/)
- [`ollama`](https://ollama.com/)

For non-technical background users, please refer [FAQ section](#key-concenpts-for-non-technical-background-users) for better understanding what those softwares are.
After installing ollama, you need to first pull llm.

Steps to prepare LLM:
- Open terminal (mac user: type `terminal` in launch pad).
- Run `ollama pull qwen3:30b-a3b` in terminal. This step will pull `qwen3:30b-a3b` from ollama and will take a while to finish. Depending on network bandwith, it took me 20min to pull. The model itself is around 20G, so be patient when downloading the model :).


## Prepare Docker Image
There two ways to prepare the docker image.
- Pull prebuild docker image
- Build docker image from source code.

### Pull Prebuild Docker Image.
TODO: build deploy ready docker image for fast deployment.

### Build Docker Image from Source Code.
Follow below steps to build docker image from source code.
- clone repo to local.
- Download model weight: cd to local repo and run `python tools/download_model.py`. The python script will use hugging-face cli to download pdf parser and embedding model weights from hugging-face.
- Build docker image. Run `docker build --build-arg NEED_MIRROR=1 -f Dockerfile -t tiny_rag:dev` to build docker image. A docker image with name `tiny_rag:dev` will be built. You can use command `docker image ls` to check images.


## Change Configuration
- Configuration file: Tiny rag container requires `.env` and `docker-compose-macos.yml` file. Download them to local.
- Configuration item explanation:
    - `IMAGE`: docker image version to use.
    - `HOST_RAG_FILE_DIR`: host directory for saving knowledge file. Tiny RAG will monitor this directory for any content change. If new file found / file deleted, Tiny RAG will automatically trigger job to parse / remove content. Subdirectory is ignored in monitoring, which means if you put file under `HOST_RAG_FILE_DIR/some_dir` the file will be ignored.
    - `HOST_RAG_LOG_DIR`: host directory for saving Tiny RAG logs.
    - `CHAT_MODEL_URL`: local ollama host url. The url parts defaut to `http://host.docker.internal` because Tiny RAG is accessing ollama from docker container.
    - `CHAT_MODEL_NAME`: model name used for chat. 


## Start backend container
Run ` docker compose -f docker-compose-macos.yml up -d` to start Tiny RAG container.

## Add knowledge base file
Put your files to `HOST_RAG_FILE_DIR`, Tiny RAG will begin parsing the files.


## Start chat
Run `docker exec -it tiny_rag_server python chat.py` to start chat with knowledge base.



# FAQ
## Key Concenpts for Non-technical Background Users
- docker
- host
- container
- image
- absolute / relative file path
- ollama



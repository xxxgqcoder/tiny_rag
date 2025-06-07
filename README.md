# TinyRAG
Tiny RAG is a simplified LLM base RAG, which act a personal knowledge assistance
and focuses only on core logic of an RAG system.

Key features 
1. Local directory base kowledge base management. All knowbase files are stored in local file folder. Tiny RAG monitors local file directory. Once file created / deleted, Tiny RAG will automatically digest and delete corresponding content.
2. Advanced PDF parser library used. Tiny RAG adopts [`MinerU`](https://github.com/opendatalab/MinerU) for pdf parsing.
3. User [`Milvus`](https://milvus.io/) as vector db for fast and hybird (dense vector matching and key-word combined) retrival.
4. Leverage local [`ollama`](https://ollama.com/) LLM for conversation.
5. Deploy through [`docker`](https://www.docker.com/).



# Usage

## Install docker and ollama
- docker
- ollama

## Docker image
- Build docker image from source code.
- Pull prebuild docker image


## Configuration
- Donwload compose file and .env file
- Local knowledge base file directory.
- Ollama chat server.


## Start backend container
- `docker compose -f`



## Add knowledge base file
- how to manage knowledge base file


## Start chat
- Terminal chat
- Restore previous chat



# FAQ
Key concenpts for non-technical background users
- docker
- host
- container
- image
- absolute / relative file path


# Tiny RAG
Tiny RAG 是一个简化版的RAG系统，作为个人知识助手，仅专注于RAG系统的核心逻辑。

关键特性
1. 基于本地目录的知识文件管理。所有知识库文件都存储在本地文件夹中。Tiny RAG会监控本地文件目录，当文件被创建/删除时，会自动处理或删除对应内容。
2. 使用先进的PDF解析库。Tiny RAG采用 [`MinerU`](https://github.com/opendatalab/MinerU) 进行PDF解析。
3. 使用 [`Milvus`](https://milvus.io/) 作为向量数据库，实现快速且混合（密集向量与关键词匹配结合）的内容检索。
4. 利用本地 [`ollama`](https://ollama.com/) LLM 进行对话。
5. 通过 [`docker`](https://www.docker.com/) 部署。

# 如何部署

## 安装 Docker 和 Ollama
要部署此项目，需要安装以下两种软件。
- [`docker`](https://www.docker.com/)
- [`ollama`](https://ollama.com/)

对于没有技术背景的用户，请参考 [FAQ部分](#非技术背景用户的关键词概念) 以更好地理解这些软件的作用。
安装完ollama后，需要先拉取一个llm模型。

拉取LLM的步骤：
- 打开终端（Mac用户：在启动台输入`terminal`）。
- 在终端运行 `ollama pull qwen3:30b-a3b`。此步骤会从ollama拉取`qwen3:30b-a3b`模型，需要较长时间完成。根据网络带宽不同，我这里耗时约20分钟。模型本身约20G，下载时请保持耐心：)。

## 准备Docker镜像
有两种方式准备Docker镜像：
- 拉取预构建的Docker镜像
- 从源代码构建Docker镜像

### 拉取预构建的Docker镜像
一个预构建镜像 xxxggxyz/tiny_rag:0.0.4 已经构建并推送到 Docker Hub。如果您希望快速尝试，可以跳过镜像构建步骤，直接修改配置阶段，进入 启动后端容器 部分。

### 从源代码构建Docker镜像
按照以下步骤从源代码构建Docker镜像：
- 克隆仓库到本地。
- 下载模型权重：进入本地仓库目录并运行 `python tools/download_model.py`。该Python脚本会使用hugging-face cli从hugging-face下载PDF解析器和嵌入模型权重。你可能需要运行 `huggingface-cli login --token $YOUR_HUGGINGFACE_TOKEN` 来获取访问权限。
- 构建Docker镜像。运行 `docker build --build-arg NEED_MIRROR=1 -f Dockerfile -t tiny_rag:dev` 来构建Docker镜像。将生成名为 `tiny_rag:dev` 的镜像。可以通过 `docker image ls` 命令查看。

## 修改配置
- 配置文件：Tiny RAG容器需要 `env` 和 `docker-compose-macos.yml` 文件。将它们下载到本地。
- 配置项说明：
    - `IMAGE`: 要使用的Docker镜像版本。
    - `HOST_RAG_FILE_DIR`: 保存知识文件的主机目录。Tiny RAG会监控此目录中的内容变化。发现新文件/删除文件时，会自动触发解析/删除任务。监控时会忽略子目录，即如果将文件放在 `HOST_RAG_FILE_DIR/some_dir` 下，该文件将被忽略。
    - `HOST_RAG_LOG_DIR`: 保存Tiny RAG日志的主机目录。
    - `CHAT_MODEL_URL`: 本地ollama主机地址。由于Tiny RAG从Docker容器访问ollama，默认URL部分为 `http://host.docker.internal`。
    - `CHAT_MODEL_NAME`: 用于对话的模型名称，设置为通过ollama拉取的LLM模型名称，此处为 `qwen3:30b-a3b`。

## 启动后端容器
运行 `docker compose -f docker-compose-macos.yml --env-file env up -d` 启动Tiny RAG容器。

## 添加知识库文件
将文件放入 `HOST_RAG_FILE_DIR`，Tiny RAG将开始解析文件。

## 启动聊天
运行 `docker exec -it tiny_rag_server python chat.py` 与知识库进行聊天。

# FAQ
## 非技术背景用户的关键词概念
- docker
- host
- container
- image
- 绝对/相对文件路径
- ollama
import argparse
import logging
import os
import pickle
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from io import TextIOWrapper

from common.config import TinyRAGConfig
from common.data import Content, ContentType
from common.utils import estimate_token_num, load_base64_image, time_it
from common.logger import Logger
from rag.llm import get_chat_model, get_vision_model

line_breaker = "\n\n"

lang_mapping = {"en": "英语", "zh": "中文"}
PROMPT_TRANSLATE = """
你是一个论文翻译助手，请将下面的{src_lang}内容翻译成{target_lang}。

下面是需要翻译的内容：

{content}

上面是需要翻译的内容

注意：
- 如果要翻译的内容为引用文献、算法伪代码、代码、人名，则不需要翻译，直接返回原文
- 你只需要输出最终翻译结果，不要输出逐步思考过程。

现在开始逐步思考
"""

PROMPT_SUMMARY = """
你是一个论文阅读助手，阅读下面的{src_lang}论文内容，并完成指令。

下面是论文内容


{content}


上面是论文内容

指令：使用{target_lang}语言，总结{src_lang}论文内容，总结的内容需要包括论文主要创新点。

注意：
- 忽略论文引用文献部分内容，只总结论文正文部分。
- 你只需要输出最终总结结果，不要输出逐步思考过程。

现在开始逐步思考
"""


# ------------------------------------------------------------------------------
def safe_strip(raw: str) -> str:
    if raw is None or len(raw) == 0:
        return ""
    raw = str(raw)
    return raw.strip()


def is_empty(text: str) -> bool:
    text = safe_strip(text)
    if text is None:
        return True
    if len(text) == 0:
        return True
    if text == "[]":
        return True
    return False


def relative_md_image_path(sys_image_folder: str, img_name: str) -> str:
    """
    md image can only be corrct displayed when saved to same folder of the md file,
    reformatted inserted image path for better display.
    """
    return f"![[{os.path.join(os.path.basename(sys_image_folder), img_name)}]]" + line_breaker


def post_text_process(text: str) -> str:
    # strip space around $
    p = r" *(\$) *"
    text = re.sub(p, r"\1", text)

    return text


def ensure_utf(text: str) -> str:
    if text is None:
        return text
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    return text


# ------------------------------------------------------------------------------
# job executor
job_executor = None


def get_job_executor() -> ProcessPoolExecutor:
    global job_executor
    if job_executor is None:
        job_executor = ProcessPoolExecutor(max_workers=1)
    return job_executor


def parse_pdf_job(file_path: str, temp_content_dir: str) -> None:
    from parse.pdf_parser import PDFParser

    Logger.info(f"Begin to process file: {file_path}")
    try:
        parser = PDFParser()
        content_list: list[Content] = parser.parse(file_path)
    except Exception as e:
        Logger.error(f"Parse failed:\n{e}")
        return

    Logger.info(f"Parsed {len(content_list)} contents from {file_path}")
    # HACK: hard coded parsed file path.
    pickle_content_path = os.path.join(temp_content_dir, "content_list.pickle")
    os.makedirs(temp_content_dir, exist_ok=True)
    with open(pickle_content_path, "wb") as f:
        pickle.dump(content_list, f)
        Logger.info(f"Saved content list to {pickle_content_path}")


@time_it(prefix="parse_pdf")
def parse_pdf(file_path: str, temp_content_dir: str) -> list[Content]:
    job_executor = get_job_executor()
    job_executor.submit(
        parse_pdf_job,
        file_path=file_path,
        temp_content_dir=temp_content_dir,
    )
    try:
        shutil.rmtree(temp_content_dir)
    except Exception as e:
        Logger.error(f"Remove temp content dir {temp_content_dir} failed:\n{e}")
    try:
        job_executor.shutdown(wait=True)
    except Exception as e:
        Logger.info(e)
        os._exit(0)

    Logger.info("PDF parse job done")

    # parse returned content
    # HACK: hard coded parsed file path.
    pickle_content_path = os.path.join(temp_content_dir, "content_list.pickle")
    Logger.info(f"Loading content list from {pickle_content_path}")
    with open(pickle_content_path, "rb") as f:
        content_list = pickle.load(f)
    Logger.info(f"Loaded {len(content_list)} content from {pickle_content_path}")

    return content_list


# ------------------------------------------------------------------------------
# save parsed content
def save_parsed_content(md_writer: TextIOWrapper, content_list: list[Content]) -> None:
    md_writer.write("# " + "=" * 4 + "  Original Content  " + "=" * 4 + line_breaker)
    vision_model = get_vision_model()

    for i, content in enumerate(content_list):
        lines = ""
        if content.content_type == ContentType.TEXT:
            lines += content.content + line_breaker
        elif content.content_type in [ContentType.IMAGE, ContentType.TABLE]:
            # HACK: rewrite image path as relative path. Specific to Obsidian.
            if content.content_url:
                img_name = os.path.basename(content.content_url)
                md_img_path = relative_md_image_path(
                    sys_image_folder=os.path.dirname(content.content_url), img_name=img_name
                )
                # image description
                img_content = load_base64_image(content.content_url)
                img_description = vision_model.image_chat(
                    prompt="summarize what you see in the picture",
                    image_content=img_content,
                )
                lines += (
                    md_img_path + line_breaker + f"[[{line_breaker}{img_description}{line_breaker}]]" + line_breaker
                )

            lines += content.extra_description + line_breaker
        else:
            pass
        lines = post_text_process(lines)
        md_writer.write(lines)
        md_writer.flush()


# ------------------------------------------------------------------------------
# translate func
def translate_text_content(text: str) -> str:
    chat_model = get_chat_model()
    if is_empty(text):
        return ""

    max_byte_len = 16 * 1024
    full_result = ""
    for i in range(0, len(text), max_byte_len):
        Logger.info(f"Processing segment {i}")
        segment = text[i : i + max_byte_len]

        formatted_prompt = PROMPT_TRANSLATE.format(
            src_lang=src_lang,
            target_lang=target_lang,
            content=segment,
        )
        ret = chat_model.instant_chat(prompt=formatted_prompt)

        full_result += ret

    return full_result


@time_it(prefix="translate_content")
def translate_content(md_writer: TextIOWrapper, content_list: list[Content]) -> None:
    """
    Translate contents.
    Args:
    - content_list: a list of content.
    """
    Logger.info(f"Total {len(content_list)} contents")

    md_writer.write("# " + "=" * 4 + "  Translated Content  " + "=" * 4 + line_breaker)

    i = 0
    max_content_num = 20
    while i < len(content_list):
        # image & table
        if content_list[i].content_type in [ContentType.TABLE, ContentType.IMAGE]:
            Logger.info(f"Translating content {i}, type: {content_list[i].content_type}")
            # save images resources
            if content_list[i].content_url:
                img_name = os.path.basename(content_list[i].content_url)
                img_path = relative_md_image_path(TinyRAGConfig.parser_config.asset_save_dir, img_name)
                md_writer.write(img_path + line_breaker)

            # translate description
            translated = translate_text_content(content_list[i].extra_description)
            translated = post_text_process(translated)
            Logger.info(f"Translated content:\n{translated}")
            md_writer.write(translated + line_breaker)

            i = i + 1
            continue

        # text
        j = i + 1
        while j < len(content_list) and j - i < max_content_num and content_list[j].content_type == ContentType.TEXT:
            j += 1
        Logger.info(f"Translating content {i} to {j - 1}, type: {content_list[i].content_type}")
        content = "\n".join([content.content for content in content_list[i:j]])
        content = ensure_utf(content)
        Logger.info(f"Content to translate:\n{content}")
        translated = translate_text_content(content)
        translated = post_text_process(translated)
        Logger.info(f"Tranlated content:\n{translated}")
        md_writer.write(translated + line_breaker)

        i = j

    md_writer.flush()


# ------------------------------------------------------------------------------
# summary func
@time_it(prefix="summary_content")
def summary_content(md_writer: TextIOWrapper, content_list: list[Content]) -> None:
    global src_lang, target_lang
    Logger.info(f"Summary_content, src_lang={src_lang}, target_lang={target_lang}")
    chat_model = get_chat_model()

    full_content = ""
    for content in content_list:
        if content.content_type == ContentType.TEXT:
            full_content += content.content + line_breaker
        elif content.content_type in [ContentType.IMAGE, ContentType.TABLE]:
            full_content += content.extra_description + line_breaker
        else:
            Logger.info(f"Unrecognized content: {content}")

    Logger.info(f"Full content length: {len(full_content)}")
    token_num, _ = estimate_token_num(full_content)
    Logger.info(f"Esitmated full content token num: {token_num}")
    if token_num > TinyRAGConfig.max_context_token_num:
        ratio = float(TinyRAGConfig.max_context_token_num) / token_num
        Logger.info(f"Truncate full content by ratio: {ratio}, original length: {len(full_content)}")
        full_content = full_content[: int(len(full_content) * ratio)]

    full_content = ensure_utf(full_content)

    formatted_promt = PROMPT_SUMMARY.format(
        src_lang=src_lang,
        target_lang=target_lang,
        content=full_content,
    )
    Logger.info(f"Formatted prompt:\n{formatted_promt}")

    summary = chat_model.instant_chat(prompt=formatted_promt)
    summary = post_text_process(summary)
    Logger.info(f"Content summary:\n{summary}")

    # save
    md_writer.write("# " + "=" * 4 + "  Paper Summary  " + "=" * 4 + line_breaker)
    md_writer.write(summary + line_breaker)
    md_writer.write("-" * 4 + line_breaker)
    md_writer.flush()


step_func = {
    "original": save_parsed_content,
    "summary": summary_content,
    "translate": translate_content,
}


@time_it(prefix="process pipeline")
def process(
    file_path: str,
    temp_content_dir: str,
    final_md_file_save_dir: str,
    steps: list[str] = ["summary", "original", "translate"],
) -> None:
    """
    Process pdf file

    Args:
    - file_path: path to file.
    - temp_content_dir: temorary output directory.
    - magic_config_path: path to magic pdf parser config.
    - final_md_file_save_dir: folder for saving final md file.
    """
    Logger.info(f"Processing started, required steps: {steps}")

    os.makedirs(temp_content_dir, exist_ok=True)
    name_without_suff = os.path.basename(file_path).rsplit(".", 1)[0]
    Logger.info(f"File name without out suffix: {name_without_suff}")

    # parse pdf
    content_list = parse_pdf(file_path=file_path, temp_content_dir=temp_content_dir)

    # md writer
    md_file_path = os.path.join(final_md_file_save_dir, f"{name_without_suff}.md")
    Logger.info(f"md file path: {md_file_path}")
    with open(md_file_path, "w") as md_writer:
        md_writer.write(f"{name_without_suff}" + line_breaker)

        # apply step functions
        for step in steps:
            Logger.info(f"Processing step: {step}")
            step = step.strip()
            if step not in step_func:
                Logger.info(f"Step {step} not configured, ignore")
                continue
            func = step_func[step]
            func(md_writer=md_writer, content_list=content_list)

    Logger.info(f"Parsed markdown saved to {md_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example of argparse usage.")

    parser.add_argument("--file_path", help="path to pdf file")
    parser.add_argument("--temp_content_dir", help="path to temp content folder", default="./tmp/parsed_asset")
    parser.add_argument("--final_md_file_save_dir", help="final md file save folder", default=".")
    parser.add_argument("--src_lang", help="source paper language", default="en")
    parser.add_argument("--target_lang", help="translate target language", default="zh")
    parser.add_argument("--steps", help="required steps", default="summary,original,translate")

    args = parser.parse_args()

    temp_content_dir = os.path.realpath(args.temp_content_dir)
    Logger.info(f"Temp content dir: {temp_content_dir}")

    final_md_file_save_dir = os.path.realpath(args.final_md_file_save_dir)
    Logger.info(f"Final md save folder: {final_md_file_save_dir}")

    src_lang = lang_mapping[args.src_lang]
    Logger.info(f"Source language: {src_lang}")

    target_lang = lang_mapping[args.target_lang]
    Logger.info(f"Target language: {target_lang}")

    Logger.info(f"Processing file: {os.path.basename(args.file_path)}")

    process(
        file_path=args.file_path,
        temp_content_dir=temp_content_dir,
        final_md_file_save_dir=final_md_file_save_dir,
        steps=args.steps.split(","),
    )

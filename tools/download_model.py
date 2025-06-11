import os
import shutil
import requests
import json
from typing import Any

from huggingface_hub import snapshot_download


def download_json(url: str) -> Any:
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def download_bge_m3_model(project_dir: str):
    patterns = [
        "*.pt",
        "*.json",
        "*.bin",
        "*.model",
    ]
    model_dir = snapshot_download(
        'BAAI/bge-m3',
        allow_patterns=patterns,
        ignore_patterns=['*onnx*'],
    )
    print(f'donwloaded model_dir is: {model_dir}')

    # copy model
    target_dir = os.path.join(project_dir, 'assets/bge-m3/models')
    shutil.copytree(
        src=model_dir,
        dst=target_dir,
        dirs_exist_ok=True,
    )
    print(f'copy model from {model_dir} to {target_dir}')

    # save json config
    config_file_name = 'bge-m3.json'
    config_file = os.path.join(project_dir, "assets/bge-m3", config_file_name)
    config = {
        "model_name_or_path": "<project_root_dir>/assets/bge-m3/models",
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print(f'MinerU config save to {config_file}')


def download_mineru_model(project_dir: str):
    # part 1
    mineru_patterns = [
        "models/Layout/YOLO/*",
        "models/MFD/YOLO/*",
        "models/MFR/unimernet_hf_small_2503/*",
        "models/OCR/paddleocr_torch/*",
    ]
    model_dir = snapshot_download(
        'opendatalab/PDF-Extract-Kit-1.0',
        allow_patterns=mineru_patterns,
    )
    model_dir = model_dir + '/models'
    print(f'donwloaded model_dir is: {model_dir}')

    # layout reader
    layoutreader_pattern = [
        "*.json",
        "*.safetensors",
    ]
    layoutreader_model_dir = snapshot_download(
        'hantian/layoutreader',
        allow_patterns=layoutreader_pattern,
    )
    print(f'donwloaded layoutreader_model_dir is: {layoutreader_model_dir}')

    # copy model
    target_dir = os.path.join(project_dir, 'assets/MinerU/models')
    shutil.copytree(
        src=model_dir,
        dst=target_dir,
        dirs_exist_ok=True,
    )
    print(f'copy model from {model_dir} to {target_dir}')

    target_dir = os.path.join(project_dir,
                              'assets/MinerU/layout_reader_models')
    shutil.copytree(
        src=layoutreader_model_dir,
        dst=target_dir,
        dirs_exist_ok=True,
    )
    print(f'copy model from {layoutreader_model_dir} to {target_dir}')

    # download config json
    json_url = 'https://github.com/opendatalab/MinerU/raw/master/magic-pdf.template.json'
    config_file_name = 'magic-pdf.json'
    config_file = os.path.join(project_dir, "assets/MinerU", config_file_name)

    # <project_root_dir> will be replaced by real directory when deployed.
    json_modification = {
        'models-dir': "<project_root_dir>/assets/MinerU/models",
        'layoutreader-model-dir':
        f"<project_root_dir>/assets/MinerU/layout_reader_models",
        "consecutive_block_num": 8,
        "block_overlap_num": 3,
    }
    data = download_json(json_url)
    for key, value in json_modification.items():
        data[key] = value
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f'MinerU config save to {config_file}')


if __name__ == '__main__':
    file_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    project_dir = os.path.realpath(file_dir + "/..")
    print(f'project directory: {project_dir}')

    download_mineru_model(project_dir)
    print(f'finish downloading MinerU model')

    download_bge_m3_model(project_dir)
    print(f'finish downloading BGE-M3 model')
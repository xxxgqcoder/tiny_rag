import os
import shutil
import requests
import json
from typing import Any
from pathlib import Path

from huggingface_hub import snapshot_download as hf_snapshot_download
from modelscope import snapshot_download as ms_snapshot_download
from mineru.utils.enum_class import ModelPath


def download_model(relative_path: str, repo_mode='pipeline') -> str:
    model_source = os.getenv('MINERU_MODEL_SOURCE', "huggingface")

    repo_mapping = {
        'pipeline': {
            'huggingface': ModelPath.pipeline_root_hf,
            'modelscope': ModelPath.pipeline_root_modelscope,
            'default': ModelPath.pipeline_root_hf
        },
        'vlm': {
            'huggingface': ModelPath.vlm_root_hf,
            'modelscope': ModelPath.vlm_root_modelscope,
            'default': ModelPath.vlm_root_hf
        },
    }

    if repo_mode not in repo_mapping:
        raise ValueError(f"Unsupported repo_mode: {repo_mode}, must be 'pipeline' or 'vlm'")

    repo = repo_mapping[repo_mode].get(model_source, repo_mapping[repo_mode]['default'])

    if model_source == "huggingface":
        snapshot_download = hf_snapshot_download
    elif model_source == "modelscope":
        snapshot_download = ms_snapshot_download
    else:
        raise ValueError(f"unknown repo type: {model_source}")

    cache_dir = None

    if repo_mode == 'pipeline':
        relative_path = relative_path.strip('/')
        cache_dir = snapshot_download(repo, allow_patterns=[relative_path, relative_path + "/*"])
    elif repo_mode == 'vlm':
        if relative_path == "/":
            cache_dir = snapshot_download(repo)
        else:
            relative_path = relative_path.strip('/')
            cache_dir = snapshot_download(repo, allow_patterns=[relative_path, relative_path + "/*"])

    if not cache_dir:
        raise FileNotFoundError(f"Failed to download model: {relative_path} from {repo}")
    return cache_dir


def download_json(url: str) -> Any:
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def download_mineru_model(project_dir: str):
    # donwnload model
    model_paths = [
        ModelPath.doclayout_yolo,
        ModelPath.yolo_v8_mfd,
        ModelPath.unimernet_small,
        ModelPath.pytorch_paddle,
        ModelPath.layout_reader,
        ModelPath.slanet_plus,
    ]
    downloaded_model_dir = ""
    for model_path in model_paths:
        print(f"downloading model from: {model_path}")
        downloaded_model_dir = download_model(model_path, repo_mode='pipeline')
    print(f'donwloaded model path: {downloaded_model_dir}')

    # parse repo name
    hf_cache_dir = os.path.join(os.path.expanduser('~'), '.cache/huggingface/hub')
    relative_model_dir = downloaded_model_dir[len(hf_cache_dir):].lstrip('/')
    path_abs = Path(relative_model_dir)
    repo_name = path_abs.parts[0]
    print(f'downloaded repo name: {repo_name}')

    # copy model
    target_dir = os.path.join(project_dir, 'assets/MinerU/', repo_name)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    shutil.copytree(
        src=os.path.join(hf_cache_dir, repo_name),
        dst=target_dir,
        dirs_exist_ok=True,
        symlinks=True,
    )
    print(f'copy model from {os.path.join(hf_cache_dir, repo_name)} to {target_dir}')

    # modify json config file
    json_url = 'https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/mineru.template.json'
    config_file_name = 'magic-pdf.json'
    config_filep_path = os.path.join(project_dir, "assets/MinerU", config_file_name)
    json_modification = {
        'models-dir': {
            "pipeline": "<project_root_dir>/assets/MinerU/",
        },
        "consecutive_block_num": 8,
        "block_overlap_num": 3,
    }
    data = download_json(json_url)

    for key, value in json_modification.items():
        if key in data:
            if isinstance(data[key], dict):
                data[key].update(value, )
            else:
                data[key] = value

    with open(config_filep_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        print(f'save modified config file to path: {config_filep_path}')


def download_bge_m3_model(project_dir: str):
    from huggingface_hub import snapshot_download

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


if __name__ == '__main__':
    file_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    project_dir = os.path.realpath(file_dir + "/..")
    print(f'project directory: {project_dir}')

    download_mineru_model(project_dir)
    print(f'finish downloading MinerU model')

    download_bge_m3_model(project_dir)
    print(f'finish downloading BGE-M3 model')

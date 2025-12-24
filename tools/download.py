# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for downloading pre-trained PixArt models
"""
from torchvision.datasets.utils import download_url
import torch
import os
import argparse


pretrained_models = {'PixArt-XL-2-512x512.pth', 'PixArt-XL-2-1024-MS.pth'}
vae_models = {
    'sd-vae-ft-ema/config.json',
    'sd-vae-ft-ema/diffusion_pytorch_model.bin'
}
t5_models = {
    't5-v1_1-xxl/config.json', 't5-v1_1-xxl/pytorch_model-00001-of-00002.bin',
    't5-v1_1-xxl/pytorch_model-00002-of-00002.bin', 't5-v1_1-xxl/pytorch_model.bin.index.json',
    't5-v1_1-xxl/special_tokens_map.json', 't5-v1_1-xxl/spiece.model',
    't5-v1_1-xxl/tokenizer_config.json',
}
# 读取环境变量 HF_ENDPOINT，默认使用 hf-mirror 镜像
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")


def find_model(model_name):
    """
    Finds a pre-trained G.pt model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:
        return download_model(model_name)
    assert os.path.isfile(model_name), f'Could not find PixArt checkpoint at {model_name}'
    return torch.load(model_name, map_location=lambda storage, loc: storage)


# 新增导入 requests 库
import requests
from tqdm import tqdm


def download_with_resume(url, save_path):
    """支持断点续传的下载函数"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 检查是否已下载部分文件
    file_size = os.path.getsize(save_path) if os.path.exists(save_path) else 0
    headers = {"Range": f"bytes={file_size}-"} if file_size > 0 else {}

    response = requests.get(url, headers=headers, stream=True, timeout=30)
    total_size = int(response.headers.get("content-length", 0)) + file_size

    if response.status_code == 416:  # 已下载完成
        return
    elif response.status_code != 206 and response.status_code != 200:
        raise Exception(f"Download failed: {response.status_code}")

    with open(save_path, "ab") as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            initial=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


# 然后修改 download_model 和 download_other 函数中的下载逻辑
def download_model(model_name):
    assert model_name in pretrained_models
    local_path = f'output/pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('output/pretrained_models', exist_ok=True)
        web_path = f'{HF_ENDPOINT}/PixArt-alpha/PixArt-alpha/resolve/main/{model_name}'
        download_with_resume(web_path, local_path)  # 替换为新函数
    return torch.load(local_path, map_location=lambda storage, loc: storage)


def download_other(model_name, model_zoo, output_dir):
    assert model_name in model_zoo
    local_path = os.path.join(output_dir, model_name)
    if not os.path.isfile(local_path):
        os.makedirs(output_dir, exist_ok=True)
        web_path = f'{HF_ENDPOINT}/PixArt-alpha/PixArt-alpha/resolve/main/{model_name}'
        print(f"Downloading from: {web_path}")
        download_with_resume(web_path, local_path)  # 替换为新函数

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_names', nargs='+', type=str, default=pretrained_models)
    args = parser.parse_args()
    model_names = args.model_names
    model_names = set(model_names)

    # Download PixArt checkpoints
    for t5_model in t5_models:
        download_other(t5_model, t5_models, 'output/pretrained_models/t5_ckpts')
    for vae_model in vae_models:
        download_other(vae_model, vae_models, 'output/pretrained_models/')
    for model in model_names:
        download_model(model)    # for vae_model in vae_models:
    print('Done.')

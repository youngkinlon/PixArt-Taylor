# PixArt-Taylor: Unofficial Implementation of Taylor-Series Approximation

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Beta-orange)
![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)

This repository provides an unofficial implementation of **Taylor-series approximation** for the PixArt-XL-2 diffusion model. By utilizing historical caches and derivative estimations, this method aims to accelerate inference by approximating heavy attention layers.

---

## 🛠 1. Environment Configuration
We recommend using **Conda** to manage your Python environment.
- Python >= 3.9 (Recommend to use Anaconda or Miniconda)
- PyTorch >= 1.13.0+cu11.7
```bash
conda create -n pixart_taylor python=3.9
conda activate pixart_taylor
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url 
https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
- notice : Some libraries have been updated recently and may cause compatibility issues. If you encounter errors, please downgrade the packages to the versions specified above.
## quick start
### step1：download pre_trained weights
Use the provided script to download the necessary model checkpoints (PixArt, VAE, and T5):
```bash
cd tools
# Download the 512x512 model weights
python download.py --model_names PixArt-XL-2-512x512.pth
```
### Step 2: Inference (Generate Image)
Run the following command to generate an image. Make sure to replace the path arguments with your actual local file addresses.
```bash
python generat_native.py \
  --prompt "A biological illustration of a magical forest with glowing mushrooms, highly detailed, digital art" \
  --image_size 512 \
  --model_path "your/local/path/to/PixArt-XL-2-512x512.pth" \
  --vae_path "your/local/path/to/sd-vae-ft-ema" \
  --t5_path "your/local/path/to/t5_ckpts" \
  --steps 20 \
  --output taylor_test.png
```
- TaylorSeer Arguments: max_order:1,interval:3 
<div id="dreambooth" style="display: flex; justify-content: center;">
  <img src="taylor_test.png" width="46%" style="margin: 5px;">
</div>


## Advanced Configuration
If you need to adjust the optimization parameters or cache settings, please modify the following configuration file:

📂 Path: ./diffusion/model/nets/cache_functions/cache_init.py

You can tune the cache initialization logic here to observe its impact on inference performance.


## ⚖️ License
This project is licensed under the Apache License 2.0.

## 🙏 Acknowledgements
This work is a reproduction based on the [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha) and [Taylor-Seer](https://github.com/chen-yang-98/Taylor-Seer) frameworks. Special thanks to the original authors for their outstanding open-source contributions.
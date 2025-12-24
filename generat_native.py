import os
import torch
import argparse
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL

# 导入 PixArt 官方组件
from diffusion.model.utils import prepare_prompt_ar
from diffusion import DPMS
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2
from diffusion.model.t5 import T5Embedder
from diffusion.data.datasets import ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True, help="输入提示词")
    parser.add_argument('--image_size', default=512, type=int, choices=[512, 1024])
    parser.add_argument('--model_path', default='tools/output/pretrained_models/PixArt-XL-2-512x512.pth', type=str)
    parser.add_argument('--vae_path', default='tools/output/pretrained_models/sd-vae-ft-ema', type=str)
    parser.add_argument('--t5_path', default='tools/output/pretrained_models/t5_ckpts', type=str)
    parser.add_argument('--cfg_scale', default=4.5, type=float)
    parser.add_argument('--steps', default=20, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output', default='result_native.png', type=str)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    # 1. 加载模型 (这里会触发你的 PixArt __init__)
    print(f"正在加载 PixArt 模型...")
    latent_size = args.image_size // 8
    lewei_scale = {512: 1, 1024: 2}

    if args.image_size == 512:
        model = PixArt_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size]).to(device)
    else:
        model = PixArtMS_XL_2(input_size=latent_size, lewei_scale=lewei_scale[args.image_size]).to(device)

    # 加载权重
    state_dict = torch.load(args.model_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if 'pos_embed' in state_dict:
        del state_dict['pos_embed']
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(torch.float16)

    # 2. 加载 VAE 和 T5
    print("正在加载 VAE 和 T5...")
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    t5 = T5Embedder(device=device, local_cache=True, cache_dir=args.t5_path, torch_dtype=torch.float)

    # 3. 准备提示词和尺寸
    print(f"提示词: {args.prompt}")
    base_ratios = eval(f'ASPECT_RATIO_{args.image_size}_TEST')
    prompt_clean, _, hw, ar, _ = prepare_prompt_ar(args.prompt, base_ratios, device=device, show=False)

    # 针对 512 的简单尺寸处理
    if args.image_size == 512:
        hw = torch.tensor([[512, 512]], dtype=torch.float, device=device)
        ar = torch.tensor([[1.]], device=device)

    # 4. 文本编码
    caption_embs, emb_masks = t5.get_text_embeddings([prompt_clean])
    caption_embs = caption_embs.float()[:, None]
    null_y = model.y_embedder.y_embedding[None].repeat(1, 1, 1)[:, None]

    # 5. 采样过程 (这里会触发你的 PixArt forward)
    print("开始采样...")
    z = torch.randn(1, 4, latent_size, latent_size, device=device)
    model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)

    dpm_solver = DPMS(model.forward_with_dpmsolver,
                      condition=caption_embs,
                      uncondition=null_y,
                      cfg_scale=args.cfg_scale,
                      model_kwargs=model_kwargs)

    samples = dpm_solver.sample(z, steps=args.steps, order=2, skip_type="time_uniform", method="multistep")

    # 6. 解码并保存
    print("解码并保存图片...")
    samples = vae.decode(samples / 0.18215).sample
    save_image(samples, args.output, nrow=1, normalize=True, value_range=(-1, 1))
    print(f"完成！图片已保存至: {args.output}")


if __name__ == "__main__":
    main()
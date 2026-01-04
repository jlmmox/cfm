import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling
from omegaconf import OmegaConf

from fmboost.helpers import instantiate_from_config, load_model_weights, un_normalize_ims
"""
python inference.py \
  --config configs/flow400_64-128/unet-base_psu.yaml \
  --ckpt logs/<train_exp>/checkpoints/last.ckpt \
  --input path/to/degraded_image_or_folder \
  --out_dir outputs \
  --num_steps 40 \
  --method rk4 \
  --processing_res 256 \
  --align 64 \
  --dtype fp16 \
  --device cuda:0
"""

def get_dtype(dtype_str: str):
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype_str]


def resize_max_res(img: Image.Image, max_edge: int, align: int = 64) -> Tuple[Image.Image, Tuple[int, int]]:
    """Resize keeping aspect so the longer side == max_edge, then round to align."""
    ow, oh = img.size
    if max_edge < 0:
        return img, (ow, oh)
    scale = min(max_edge / ow, max_edge / oh)
    nw, nh = int(ow * scale), int(oh * scale)
    nw = max(align, round(nw / align) * align)
    nh = max(align, round(nh / align) * align)
    resized = img.resize((nw, nh), resample=Resampling.BILINEAR)
    return resized, (ow, oh)


def load_image(fp: str, processing_res: int, align: int = 64) -> Tuple[torch.Tensor, Tuple[int, int]]:
    img = Image.open(fp).convert("RGB")
    img, orig_res = resize_max_res(img, processing_res, align=align)
    arr = np.array(img, dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))  # C,H,W
    arr = arr / 127.5 - 1.0
    tensor = torch.tensor(arr)[None]  # 1,C,H,W
    return tensor, orig_res


def save_image(tensor: torch.Tensor, out_path: str, orig_res: Tuple[int, int]):
    img = un_normalize_ims(tensor.detach().cpu())  # uint8, C,H,W
    img = img[0].permute(1, 2, 0).numpy()  # H,W,C
    pil = Image.fromarray(img)
    if pil.size != orig_res:
        pil = pil.resize(orig_res, resample=Resampling.BILINEAR)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pil.save(out_path)


def list_images(path: str) -> List[str]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]
    if os.path.isdir(path):
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(path, "**", ext), recursive=True))
        return sorted(files)
    return [path]


def infer_one(model, img_path: str, device: torch.device, args):
    lres_ims, orig_res = load_image(img_path, args.processing_res, align=args.align)
    lres_ims = lres_ims.to(device)

    # determine target sizes
    im_size = lres_ims.shape[-1]
    has_first_stage = getattr(model, "first_stage", None) is not None
    if has_first_stage:
        z_hr_size = im_size // 8
    else:
        z_hr_size = im_size

    lres_z = model.encode_first_stage(lres_ims)

    sample_kwargs = {
        "num_steps": args.num_steps,
        "method": args.method,
    }

    use_autocast = device.type == "cuda" and args.dtype != "fp32"
    dtype = get_dtype(args.dtype)

    with torch.no_grad():
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=dtype):
                pred = model.predict_high_res_img(
                    lres_z=lres_z,
                    z_hr_size=z_hr_size,
                    lres_ims=lres_ims,
                    im_size=im_size,
                    sample_kwargs=sample_kwargs,
                )
        else:
            pred = model.predict_high_res_img(
                lres_z=lres_z,
                z_hr_size=z_hr_size,
                lres_ims=lres_ims,
                im_size=im_size,
                sample_kwargs=sample_kwargs,
            )

    rel = os.path.basename(img_path)
    name, ext = os.path.splitext(rel)
    out_path = os.path.join(args.out_dir, f"{name}_restored{ext}")
    save_image(pred, out_path, orig_res)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser("FMBoost inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Image file or folder")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--method", type=str, default="rk4", help="ODE/SDE method (rk4/euler/...) as supported by FlowModel")
    parser.add_argument("--processing_res", type=int, default=-1, help="Resize longer edge to this (rounded to align); -1 keeps size")
    parser.add_argument("--align", type=int, default=64, help="Round resized edge to this multiple")
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    cfg = OmegaConf.load(args.config)
    model = instantiate_from_config(cfg.model)
    model = load_model_weights(model, args.ckpt, strict=False, verbose=True)
    model.eval().to(device)

    files = list_images(args.input)
    if len(files) == 0:
        raise FileNotFoundError(f"No images found at {args.input}")

    print("-" * 40)
    print(f"Config      : {args.config}")
    print(f"Checkpoint  : {args.ckpt}")
    print(f"Input path  : {args.input}")
    print(f"Num images  : {len(files)}")
    print(f"Out dir     : {args.out_dir}")
    print(f"Num steps   : {args.num_steps}")
    print(f"Method      : {args.method}")
    print(f"Precision   : {args.dtype}")
    print(f"Device      : {device}")
    print("-" * 40)

    for fp in files:
        infer_one(model, fp, device, args)


if __name__ == "__main__":
    main()

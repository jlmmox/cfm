import torch
import random
import warnings
import importlib
import numpy as np
import torch.nn as nn
from functools import partial
from inspect import isfunction


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def un_normalize_ims(ims):
    """ Convert from [-1, 1] to [0, 255] """
    ims = ((ims * 127.5) + 127.5).clip(0, 255).to(torch.uint8)
    return ims


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def load_partial_from_config(config):
    return partial(get_obj_from_str(config['target']),**config.get('params',dict()))


def load_model_from_config(config, ckpt, verbose=False, ignore_keys=[]):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"] if 'state_dict' in pl_sd else pl_sd
    keys = list(sd.keys())
    for k in keys:
        for ik in ignore_keys:
            if ik and k.startswith(ik):
                print("Deleting key {} from state_dict.".format(k))
                del sd[k]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    print(f'Missing {len(m)} keys and unexpecting {len(u)} keys')
    # model.cuda()
    # model.eval()
    return model

def load_model_from_ckpt(model, ckpt, verbose=False, ignore_keys=[]):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"] if 'state_dict' in pl_sd else pl_sd
    keys = list(sd.keys())
    for k in keys:
        for ik in ignore_keys:
            if ik and k.startswith(ik):
                print("Deleting key {} from state_dict.".format(k))
                del sd[k]
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    print(f'Missing {len(m)} keys and unexpecting {len(u)} keys')
    # model.cuda()
    # model.eval()
    return model


def load_model_weights(model, ckpt_path, strict=True, verbose=True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if verbose:
        print("-" * 40)
        print(f"{'Loading weights':<16}: {ckpt_path}")
        print(f"{'Model':<16}: {model.__class__.__name__}")
        if "global_step" in ckpt:
            print(f"{'Global Step':<16}: {ckpt['global_step']:,}")
        print(f"{'Strict':<16}: {'True' if strict else 'False'}")
        print("-" * 40)
    sd = ckpt["state_dict"] if 'state_dict' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if len(missing) > 0:
        warnings.warn(f"Load model weights - missing keys: {len(missing)}")
        if verbose:
            print(missing)
    if len(unexpected) > 0:
        warnings.warn(f"Load model weights - unexpected keys: {len(unexpected)}")
        if verbose:
            print(unexpected)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def bool2str(b):
    return "True" if b else "False"


def resize_ims(x: torch.Tensor, size: int, mode: str = "bilinear", **kwargs):
    # for the sake of backward compatibility
    if mode in ["conv", "noise_upsampling", "decoder_features"]:
        return nn.functional.interpolate(x, size=size, mode="bilinear", **kwargs)
    # idea: blur image before down-sampling
    return nn.functional.interpolate(x, size=size, mode=mode, **kwargs)


def freeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def pad_to_multiple(x: torch.Tensor, multiple: int = 8, mode: str = 'reflect'):
    """Pad a tensor so its spatial dims (H, W) are divisible by `multiple`.

    Args:
        x: torch.Tensor with shape (B, C, H, W) or (C, H, W) or (H, W)
        multiple: int, pad so H and W are divisible by this
        mode: padding mode passed to torch.nn.functional.pad (e.g. 'reflect', 'constant')

    Returns:
        x_padded: padded tensor
        pads: tuple (pad_left, pad_top, pad_right, pad_bottom)
    """
    single = False
    if x.dim() == 2:
        # (H, W) -> (1, 1, H, W)
        x = x.unsqueeze(0).unsqueeze(0)
        single = True
    elif x.dim() == 3:
        # (C, H, W) -> (1, C, H, W)
        x = x.unsqueeze(0)
        single = True

    _, _, h, w = x.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    if pad_left + pad_right + pad_top + pad_bottom == 0:
        # no padding needed
        if single:
            return (x.squeeze(0) if x.dim() == 4 and single else x), (0, 0, 0, 0)
        return x, (0, 0, 0, 0)

    # torch.nn.functional.pad expects pad as (left, right, top, bottom)
    x_padded = nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode=mode)

    if single:
        # restore original batch/channel dims
        if x_padded.shape[0] == 1 and x_padded.shape[1] == 1:
            x_padded = x_padded.squeeze(0)
            # keep (C,H,W) if originally 3D
        else:
            x_padded = x_padded.squeeze(0)
    return x_padded, (pad_left, pad_top, pad_right, pad_bottom)


def unpad(x: torch.Tensor, pads: tuple):
    """Remove padding added by pad_to_multiple.

    Args:
        x: torch.Tensor with shape (B, C, H, W) or (C, H, W) or (H, W)
        pads: tuple (pad_left, pad_top, pad_right, pad_bottom)

    Returns:
        cropped tensor with original spatial size
    """
    pad_left, pad_top, pad_right, pad_bottom = pads

    single = False
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
        single = True
    elif x.dim() == 3:
        x = x.unsqueeze(0)
        single = True

    _, _, h, w = x.shape
    left = pad_left
    right = w - pad_right
    top = pad_top
    bottom = h - pad_bottom

    # ensure indices are within bounds
    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)

    x_cropped = x[..., top:bottom, left:right]

    if single:
        x_cropped = x_cropped.squeeze(0)
    return x_cropped

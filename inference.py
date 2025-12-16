"""
CFM 推理脚本 - 专注于图像还原（Restore）

已移除超分辨率（upscale）相关接口与推理路径，推理类仅保留用于图像还原的功能：
- 加载 VAE（编码/解码）
- 加载 Flow Matching 模型
- 可选的前向扩散噪声增强
- 使用 Flow Matching 在相同分辨率的潜在空间完成还原
"""

import os
import sys
import torch
from PIL import Image

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cfm.flow import FlowModel
from cfm.diffusion import ForwardDiffusion
from cfm.kl_autoencoder import AutoencoderKL
from cfm.helpers import exists


class CFMInference:
    """
    专注于图像还原的推理类（输入和输出分辨率相同）
    """

    def __init__(
        self,
        flow_model_ckpt: str,
        vae_ckpt: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        scale_factor: float = 0.18215,
        noising_step: int = 400,
        concat_context: bool = True,
    ):
        self.device = device
        self.scale_factor = scale_factor
        self.noising_step = noising_step
        self.concat_context = concat_context
        self.use_noise = noising_step > 0

        # 加载VAE（用于编码/解码）
        if exists(vae_ckpt):
            print(f"加载VAE模型: {vae_ckpt}")
            self.vae = AutoencoderKL(ckpt_path=vae_ckpt).to(device)
            self.vae.eval()
        else:
            print("警告: 未提供VAE checkpoint，将无法解码/编码图像")
            self.vae = None

        # 加载Flow Matching模型
        print(f"加载Flow Matching模型: {flow_model_ckpt}")
        self.flow_model = self._load_flow_model(flow_model_ckpt)
        self.flow_model.eval()

        # 初始化噪声增强模块
        if self.use_noise:
            print(f"初始化噪声增强模块 (step={noising_step})")
            self.diffusion = ForwardDiffusion()
            self.diffusion.to(device)
            self.diffusion.eval()
        else:
            self.diffusion = None

    def _load_flow_model(self, ckpt_path: str):
        """加载 Flow Matching 模型并尝试加载权重（保持原有兼容性）"""
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # 尝试从 checkpoint 或配置中恢复网络结构
        from cfm.models.unet import EfficientUNet

        # 这里构建一个默认网络结构，权重加载时会采用 strict=False
        net = EfficientUNet(
            in_channels=8 if self.concat_context else 4,
            model_channels=128,
            out_channels=4,
            num_res_blocks=3,
            attention_resolutions=[8, 16],
            dropout=0.0,
            channel_mult=[1, 2, 4, 8],
            conv_resample=True,
            dim_head=64,
            num_heads=4,
            use_linear_attn=False,
            use_scale_shift_norm=True,
            pool_factor=-1,
        )

        flow_model = FlowModel(net_cfg=net, schedule="linear", sigma_min=0.0)

        # 加载权重（兼容 Lightning 格式或纯 state_dict）
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            model_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_key = k[6:]
                    model_state_dict[new_key] = v
                elif k.startswith("ema_model.model."):
                    new_key = k[16:]
                    model_state_dict[new_key] = v
            if len(model_state_dict) > 0:
                try:
                    flow_model.load_state_dict(model_state_dict, strict=False)
                    print("成功加载Flow Matching模型权重")
                except Exception as e:
                    print(f"警告: 加载权重时出错: {e}")
                    print("将使用随机初始化的模型")
            else:
                print("警告: 未找到模型权重字段，将使用随机初始化的模型")
        else:
            try:
                flow_model.load_state_dict(ckpt, strict=False)
                print("成功加载Flow Matching模型权重")
            except Exception as e:
                print(f"警告: 加载权重时出错: {e}")
                print("将使用随机初始化的模型")

        return flow_model.to(self.device)

    def encode_image(self, image: torch.Tensor):
        if self.vae is None:
            raise ValueError("VAE未加载，无法编码图像")
        with torch.no_grad():
            latent = self.vae.encode(image, normalize=True)
        return latent

    def decode_latent(self, latent: torch.Tensor):
        if self.vae is None:
            raise ValueError("VAE未加载，无法解码图像")
        with torch.no_grad():
            image = self.vae.decode(latent, denorm=True)
        return image

    def add_noise(self, x: torch.Tensor):
        if not self.use_noise or self.diffusion is None:
            return x
        with torch.no_grad():
            x_noisy = self.diffusion.q_sample(x_start=x, t=self.noising_step)
        return x_noisy

    def flow_matching_restore(self, x_source: torch.Tensor, context: torch.Tensor = None, num_steps: int = 40, method: str = "rk4"):
        """在相同分辨率潜在空间进行还原推理（用于 restore）"""
        sample_kwargs = {"num_steps": num_steps, "method": method, "cfg_scale": 1.0}
        with torch.no_grad():
            hres_z = self.flow_model.generate(x=x_source, context=context, sample_kwargs=sample_kwargs)
        return hres_z

    def restore_image(self, image: torch.Tensor, num_steps: int = 40, return_intermediates: bool = False):
        """图像还原主流程（输入输出分辨率相同）"""
        results = {}
        # 编码
        latent = self.encode_image(image)
        if return_intermediates:
            results['input_latent'] = latent.clone()
        # 噪声增强
        if self.use_noise:
            x_source = self.add_noise(latent)
        else:
            x_source = latent
        if return_intermediates:
            results['noisy_latent'] = x_source.clone()
        # 上下文（可选）
        if self.concat_context:
            context = latent
        else:
            context = None
        # Flow Matching 还原
        restored_latent = self.flow_matching_restore(x_source=x_source, context=context, num_steps=num_steps)
        if return_intermediates:
            results['restored_latent'] = restored_latent.clone()
        # 解码
        restored_image = self.decode_latent(restored_latent)
        if return_intermediates:
            results['restored_image'] = restored_image.clone()
            return restored_image, results
        else:
            return restored_image


def main():
    """命令行入口（仅保留图像还原用法）"""
    import argparse
    parser = argparse.ArgumentParser(description="CFM 图像还原推理脚本（仅支持 restore）")
    parser.add_argument("--flow_ckpt", type=str, required=True, help="Flow Matching模型checkpoint路径")
    parser.add_argument("--vae_ckpt", type=str, default=None, help="VAE模型checkpoint路径")
    parser.add_argument("--input", type=str, required=True, help="输入图像路径（用于还原）")
    parser.add_argument("--output", type=str, default="output.png", help="输出图像路径")
    parser.add_argument("--num_steps", type=int, default=40, help="Flow Matching ODE求解步数")
    parser.add_argument("--noising_step", type=int, default=400, help="噪声增强步数（-1表示不使用）")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    args = parser.parse_args()

    inferencer = CFMInference(
        flow_model_ckpt=args.flow_ckpt,
        vae_ckpt=args.vae_ckpt,
        device=args.device,
        noising_step=args.noising_step,
    )

    from torchvision import transforms
    img = Image.open(args.input).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = transform(img).unsqueeze(0).to(args.device)

    restored_image = inferencer.restore_image(image=image, num_steps=args.num_steps)
    output_image = restored_image[0].cpu()

    img_array = ((output_image.clamp(-1, 1) + 1) * 127.5).clamp(0, 255).byte()
    img_array = img_array.permute(1, 2, 0).numpy()
    Image.fromarray(img_array).save(args.output)
    print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()


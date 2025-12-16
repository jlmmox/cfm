# CFM 推理与训练模块

耦合流匹配（Coupling Flow Matching, CFM）模块，已精简为专注于图像还原任务（Restore）。

## 快速开始

### 推理（图像还原）

```python
from cfm_inference.inference import CFMInference

inferencer = CFMInference(
    flow_model_ckpt="path/to/flow_model.ckpt",
    vae_ckpt="checkpoints/sd_ae.ckpt"
)
```

### 训练

```bash
python train.py --config configs/image_restore_512.yaml --name my_restore_experiment
```

**详细文档**：
- 📖 [训练指南](docs/TRAINING_GUIDE.md) - 完整的训练流程和配置说明（已以图像还原为主）
- 📖 [VAE说明](docs/WHY_VAE.md) - 为什么训练需要VAE

## 功能特性

本模块现在专注于图像还原模式（Image Restoration）：

1. **输入**: 受损图像（低质量图像）
2. **编码**: 将图像编码到潜在空间
3. **噪声增强**: 使用前向扩散过程对潜在编码添加噪声（可选）
4. **流匹配还原**: 使用Flow Matching模型在相同分辨率的潜在空间进行还原
5. **解码**: 使用VAE解码器将还原后的潜在编码解码为图像（相同分辨率）

## 使用方法（图像还原）

```python
from cfm_inference.inference import CFMInference
from torchvision import transforms
from PIL import Image

inferencer = CFMInference(
    flow_model_ckpt="path/to/flow_model.ckpt",
    vae_ckpt="path/to/vae.ckpt",
    device="cuda",
    noising_step=400,
)

img = Image.open("input.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
image = transform(img).unsqueeze(0).to("cuda")

restored_image = inferencer.restore_image(
    image=image,
    num_steps=40,
)

# 保存结果
img_array = ((restored_image[0].cpu().clamp(-1, 1) + 1) * 127.5).clamp(0, 255).byte()
img_array = img_array.permute(1, 2, 0).numpy()
Image.fromarray(img_array).save("output.png")
```

## 目录结构（相关部分）

```
cfm_inference/
├── cfm_inference/          # 核心模块
│   ├── inference.py        # 推理脚本（图像还原）
│   ├── flow.py            # Flow Matching模型
│   ├── diffusion.py       # 噪声增强（前向扩散）
│   ├── kl_autoencoder.py  # VAE编码解码器
│   ├── helpers.py         # 辅助函数
│   ├── trainer.py         # 训练模块（图像还原）
│   ├── ema.py             # EMA模型更新
│   ├── metrics.py         # 评估指标
│   ├── dataloader.py      # 数据加载器
│   └── models/            # 网络模型
```

## 参数说明（图像还原）

- `flow_model_ckpt` (str): Flow Matching模型checkpoint路径
- `vae_ckpt` (str, optional): VAE模型checkpoint路径，用于编码/解码
- `device` (str): 设备 ('cuda' 或 'cpu')
- `noising_step` (int): 噪声增强步数，默认400（-1表示不使用噪声增强）

## 命令行示例（图像还原）

```bash
python inference.py --flow_ckpt path/to/flow.ckpt --vae_ckpt checkpoints/sd_ae.ckpt --input input.jpg --output output.png --num_steps 40
```

## 训练

使用 `configs/image_restore_512.yaml` 中的默认配置进行图像还原训练。


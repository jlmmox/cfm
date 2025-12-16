# 图像还原训练指南（精简版）

本指南专注于在本项目中进行图像还原（Image Restoration）任务的训练与准备。

目录
1. 准备工作
2. 下载与准备 VAE checkpoint
3. 数据集格式（受损-无损图像对）
4. 配置文件示例（图像还原）
5. 训练命令与注意事项
6. 验证与推理


1. 准备工作
-----------------
- Python 环境（建议使用虚拟环境）
- 安装依赖：

```bash
pip install -r requirements.txt
```

- 确保 `checkpoints/sd_ae.ckpt` 已下载并放置（见下一节）。


2. 下载与准备 VAE checkpoint
-----------------
VAE（AutoencoderKL）用于将图像编码到潜在空间并解码回图像。训练/推理时 VAE 参数会被冻结，仅用于编码/解码。

下载示例（请替换为有效链接）：

```bash
mkdir -p checkpoints
# 下载并保存为 checkpoints/sd_ae.ckpt
# wget -O checkpoints/sd_ae.ckpt <VAE_CHECKPOINT_URL>
```

在配置文件中，指定 VAE 路径：

```yaml
first_stage_cfg:
  target: cfm_inference.kl_autoencoder.AutoencoderKL
  params:
    ckpt_path: checkpoints/sd_ae.ckpt
```


3. 数据集格式（受损-无损图像对）
-----------------
训练时数据应为成对的受损（degraded）与无损（clean）图像：

```
data/
├── train/
│   ├── degraded/   # 受损图像（输入）
│   └── clean/      # 无损图像（目标）
└── val/
    ├── degraded/
    └── clean/
```

- 推荐文件名一致（例如 `image001.jpg` 在 degraded 和 clean 目录中均存在）。
- 我们提供了 `cfm_inference/cfm_inference/dataloader.py` 中的 `PairedFolderDataModule` 数据模块，会尝试自动匹配文件名并在加载时进行尺寸调整与归一化。


4. 配置文件示例（图像还原）
-----------------
以下是用于图像还原的推荐配置 `configs/image_restore_512.yaml` 中的关键字段示例：

```yaml
model:
  target: cfm_inference.trainer.TrainerFMBoost
  params:
    low_res_size: 512
    high_res_size: 512  # 与 low_res_size 相同，表示输入输出分辨率不变
    upsampling_mode: identity
    upsampling_mode_context: identity
    upsampling_mode_ca_context: identity
    start_from_noise: False
    noising_step: 400
    concat_context: True
    ca_context: False
    fm_cfg:
      target: cfm_inference.flow.FlowModel
      params:
        schedule: linear
        
        net_cfg:
          target: cfm_inference.models.unet.model.EfficientUNet
          params:
            in_channels: 8
            model_channels: 128
            out_channels: 4
            num_res_blocks: 3
            channel_mult: [1, 2, 4, 8]
            attention_resolutions: [8, 16]
    scale_factor: 0.18215
    first_stage_cfg:
      target: cfm_inference.kl_autoencoder.AutoencoderKL
      params:
        ckpt_path: checkpoints/sd_ae.ckpt
    lr: 3e-5
    ema_rate: 0.999
    use_ema_for_sampling: True

data:
  target: cfm_inference.dataloader.PairedFolderDataModule
  params:
    train_degraded_dir: data/train/degraded
    train_clean_dir: data/train/clean
    val_degraded_dir: data/val/degraded
    val_clean_dir: data/val/clean
    batch_size: 4
    num_workers: 4
    image_size: 512
```


5. 训练命令与注意事项
-----------------
基本训练命令：

```bash
python train.py --config configs/image_restore_512.yaml --name my_restore_experiment --use_wandb
```

注意：
- 确保 `low_res_size` 与 VAE 的潜在下采样尺度匹配（常见为原图 /8）。
- VAE checkpoint 在训练中被冻结，仅用于编码/解码。
- 如果训练数据为大图，请根据显存调整 batch_size 与 image_size。


6. 验证与推理
-----------------
训练过程中会在验证集上计算 MSE / PSNR / SSIM / FID 指标（如果 metric_tracker 在配置中启用）。

推理（图像还原）示例：

```python
from cfm_inference.inference import CFMInference
from torchvision import transforms
from PIL import Image

inferencer = CFMInference(
    flow_model_ckpt="path/to/flow_model.ckpt",
    vae_ckpt="checkpoints/sd_ae.ckpt",
    device="cuda",
    noising_step=400,
)

img = Image.open("input.jpg").convert("RGB")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
image = transform(img).unsqueeze(0).to("cuda")
restored_image = inferencer.restore_image(image=image, num_steps=40)

# 保存结果
img_array = ((restored_image[0].cpu().clamp(-1, 1) + 1) * 127.5).clamp(0, 255).byte()
img_array = img_array.permute(1, 2, 0).numpy()
from PIL import Image
Image.fromarray(img_array).save("output.png")
```


---

我将同时查找并删除仓库中与超分辨率直接相关的引用（包括 configs、docs、代码注释）。下面是要删除或已删除的项目清单：

- 已删除/精简：
  - `cfm_inference/inference.py` 中的超分辨率路径（已改为仅支持 restore）
  - `cfm_inference/cfm_inference/trainer.py` 中的 upsample_latent/predict_high_res_* 等方法（已移除）
  - `cfm_inference/README.md` 中的超分辨率示例（已移除）
  - `cfm_inference/docs/TRAINING_GUIDE.md` 已精简为还原指南（本文件）

- 仍保留但建议归档或删除（因为它们与超分辨率相关）：
  - `cfm_inference/configs/flow400_64-128/unet-base_psu.yaml` （包含 psu/upsampling_mode 配置）
  - `fm-boosting/` 目录（完整原始实现，含超分辨率逻辑）

如果你确认要我也删除/移动上述“仍保留”的文件（如删除 `fm-boosting/`、删除 `configs/flow400_64-128/unet-base_psu.yaml`），我可以执行这些删除操作。请确认是否继续删除这些文件，或者我只列出位置让你手动处理。


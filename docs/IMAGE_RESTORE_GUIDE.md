# 图像还原任务训练指南

专门针对**图像还原任务**（受损图像 → 无损图像）的完整训练指南。

## 任务说明

- **输入**: 受损图像（低质量、有噪声、模糊等）
- **输出**: 无损图像（高质量、清晰）
- **分辨率**: 输入输出**相同**（如512×512）

## 快速开始

### 1. 准备数据集

将你的图像按以下结构放置：

```
data/
├── train/
│   ├── degraded/          # 受损图像（输入）
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   └── clean/             # 无损图像（目标）
│       ├── image001.jpg   # 与degraded中对应，文件名相同
│       ├── image002.jpg
│       └── ...
└── val/
    ├── degraded/          # 验证集受损图像
    │   └── val001.jpg
    └── clean/             # 验证集无损图像
        └── val001.jpg
```

**重要**：
- 受损图像和无损图像应该**成对出现**
- 建议文件名相同（如 `image001.jpg` 在degraded和clean目录中都有）
- 如果文件名不同，数据加载器会尝试自动匹配（去掉_degraded等后缀）

### 2. 使用配置文件

使用提供的配置文件：`configs/image_restore_512.yaml`

这个配置文件已经设置好：
- 输入输出分辨率相同（512×512）
- 使用图像还原数据加载器
- 优化的训练参数

### 3. 开始训练

```bash
python train.py \
    --config configs/image_restore_512.yaml \
    --name image_restore_experiment \
    --use_wandb
```

## 详细配置说明

### 配置文件：`configs/image_restore_512.yaml`

```yaml
model:
  params:
    # 图像还原：输入输出分辨率相同
    low_res_size: 512
    high_res_size: 512
    
    # 上采样模式（图像还原时使用identity，因为不需要上采样）
    upsampling_mode: identity
    
    # 噪声增强（增强模型鲁棒性）
    noising_step: 400
    
    # 拼接上下文（将受损图像作为条件）
    concat_context: True

data:
  target: cfm_inference.dataloader.PairedFolderDataModule
  params:
    root: data
    batch_size: 4
    image_size: 512
    val_batch_size: 4
    num_workers: 4
    seed: 42
    random_flip: True
```

### 数据加载器

已提供现成的数据加载器 `cfm_inference/dataloader.py` 中的 `PairedFolderDataModule`，它会：
- 自动加载受损-无损图像对
- 自动匹配文件名
- 调整图像尺寸（短边缩放+裁剪）
- 归一化到[-1, 1]范围
- 自动划分验证集（如果未提供单独的验证集目录）

## 训练流程

训练时，模型会执行以下步骤：

1. **加载图像对**：
   - 从 `degraded/` 目录加载受损图像（输入）
   - 从 `clean/` 目录加载对应的无损图像（目标）

2. **编码到潜在空间**：
   - 使用VAE将受损图像编码为潜在编码
   - 使用VAE将无损图像编码为潜在编码

3. **Flow Matching训练**：
   - 在潜在空间中学习从受损图像到无损图像的映射
   - 输入输出分辨率保持不变

4. **验证**：
   - 计算MSE、PSNR、SSIM、FID等指标
   - 可视化结果（受损图像、无损图像、预测图像）

## 训练指标

训练过程中会自动计算：
- **MSE**: 均方误差（越低越好）
- **PSNR**: 峰值信噪比（越高越好）
- **SSIM**: 结构相似性指数（越高越好，0-1范围）
- **FID**: FID分数（越低越好）

## 常见问题

### Q: 文件名必须相同吗？

A: 建议相同，但如果不同，数据加载器会尝试自动匹配：
- 去掉 `_degraded`, `_damaged`, `_low` 等后缀
- 尝试不同的扩展名（.jpg, .png等）

### Q: 图像尺寸必须是512×512吗？

A: 不是，可以在配置文件中修改 `image_size` 参数。但需要同时修改：
- `low_res_size` 和 `high_res_size`（设置为相同值）
- `image_size` 在data配置中

### Q: 如何调整batch size？

A: 在配置文件的 `data.params.batch_size` 中修改。

### Q: 训练后如何使用模型？

A: 使用推理接口：

```python
from cfm_inference.inference import CFMInference

inferencer = CFMInference(
    flow_model_ckpt="logs/your_experiment/checkpoints/last.ckpt",
    vae_ckpt="checkpoints/sd_ae.ckpt"
)

# 图像还原
restored_image = inferencer.restore_image(image, num_steps=40)
```

## 完整训练示例

```bash
# 1. 创建目录
mkdir -p checkpoints data/train/degraded data/train/clean data/val/degraded data/val/clean logs

# 2. 下载VAE checkpoint
cd checkpoints
curl -L "https://www.dropbox.com/scl/fi/lvfvy7qou05kxfbqz5d42/sd_ae.ckpt?rlkey=fvtu2o48namouu9x3w08olv3o&st=vahu44z5&dl=1" -o sd_ae.ckpt
cd ..

# 3. 准备数据集
#    将受损图像放入 data/train/degraded/ 和 data/val/degraded/
#    将无损图像放入 data/train/clean/ 和 data/val/clean/

# 4. 开始训练
python train.py \
    --config configs/image_restore_512.yaml \
    --name my_restore_training \
    --use_wandb \
    --devices 1

# 5. 监控训练
tensorboard --logdir logs/
```

## 与超分辨率任务的区别

| 特性 | 超分辨率 | 图像还原 |
|------|---------|---------|
| **输入** | 低分辨率图像 | 受损图像 |
| **输出** | 高分辨率图像 | 无损图像 |
| **分辨率** | 不同（低→高） | 相同 |
| **数据集** | 仅需高分辨率图像 | 需要图像对 |
| **上采样** | 需要 | 不需要（identity） |
| **配置文件** | `unet-base_psu.yaml` | `image_restore_512.yaml` |


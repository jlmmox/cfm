import wandb
import einops
import warnings
from PIL import Image

import torch
import torch.nn as nn
from torch import Tensor
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import WandbLogger

from cfm.ema import EMA
from cfm.diffusion import ForwardDiffusion

from cfm.helpers import freeze
from cfm.helpers import resize_ims
from cfm.helpers import exists, default
from cfm.helpers import un_normalize_ims
from cfm.helpers import instantiate_from_config
from cfm.helpers import load_partial_from_config


def hres_lres_pred_grid(hr_ims, lr_ims, hr_pred):
    # resize lr_ims if necessary
    if lr_ims.shape[-1] != hr_ims.shape[-1]:
        lr_ims = resize_ims(lr_ims, hr_ims.shape[-1], mode="bilinear")
    hr_ims = einops.rearrange(hr_ims, "b c h w -> (b h) w c")
    lr_ims = einops.rearrange(lr_ims, "b c h w -> (b h) w c")
    hr_pred = einops.rearrange(hr_pred, "b c h w -> (b h) w c")
    grid = torch.cat([hr_ims, lr_ims, hr_pred], dim=1)
    # normalize to [0, 255]
    grid = un_normalize_ims(grid).cpu().numpy()
    return grid


class TrainerFMBoost(LightningModule):
    def __init__(
            self,
            fm_cfg: dict,
            low_res_size: int,
            high_res_size: int = None,              # unused
            upsampling_mode: str = "bilinear",
            upsampling_mode_context: str = None,
            upsampling_mode_ca_context: str = None,
            start_from_noise: bool = False,
            noising_step: int = -1,
            concat_context: bool = False,
            ca_context: bool = False,
            first_stage_cfg: dict = None,
            scale_factor: int = 1.0,
            lr: float = 1e-4,
            weight_decay: float = 0.,
            n_images_to_vis: int = 16,
            ema_rate: float = 0.99,
            ema_update_every: int = 100,
            ema_update_after_step: int = 1000,
            use_ema_for_sampling: bool = True,
            metric_tracker_cfg: dict = None,
            lr_scheduler_cfg: dict = None,
            log_grad_norm: bool = False,
        ):
        """
        Args:
            fm_cfg: Flow matching model config.
            low_res_size: Size of low-res images.
            upsampling_mode: Mode for up-sampling (bilinear, nearest, psu).
            upsampling_mode_context: Mode for up-sampling the concatenated
                context (if None, same as upsampling_mode).
            upsampling_mode_ca_context: Mode for up-sampling the cross-attention
                context (if None, same as upsampling_mode).
            start_from_noise: Whether to start from noise with low-res image as
                conditioning (FM) or directly from low-res image (IC-FM).
            noising_step: Forward diffusion noising step with linear schedule
                of Ho et al. Set to -1 to disable.
            concat_context: Whether to concatenate the low-res images as conditioning.
            ca_context: Whether to use cross-attention context.
            first_stage_cfg: First stage config, if None, identity is used.
            scale_factor: Scale factor for the latent space (normalize the 
                latent space, default value for SD: 0.18215).
            lr: Learning rate.
            weight_decay: Weight decay.
            n_images_to_vis: Number of images to visualize.
            ema_rate: EMA rate.
            ema_update_every: EMA update rate (every n steps).
            ema_update_after_step: EMA update start after n steps.
            use_ema_for_sampling: Whether to use the EMA model for sampling.
            metric_tracker_cfg: Metric tracker config.
            lr_scheduler_cfg: Learning rate scheduler config.
            log_grad_norm: Whether to log the gradient norm.
        """
        super().__init__()
        self.model = instantiate_from_config(fm_cfg)
        # self.model = torch.compile(self.model)            # TODO haven't fully debugged yet
        self.ema_model = EMA(
            self.model, beta=ema_rate,
            update_after_step=ema_update_after_step,
            update_every=ema_update_every,
            power=3/4.,                     # recommended for trainings < 1M steps
            include_online_model=False      # we have the online model stored here

        )
        self.use_ema_for_sampling = use_ema_for_sampling

        assert low_res_size % 8 == 0, "Low-res size must be divisible by 8 (AE)"
        self.low_res_size = low_res_size

        self.upsampling_mode = upsampling_mode
        self.upsampling_mode_context = default(upsampling_mode_context, upsampling_mode)
        self.upsampling_mode_ca_context = default(upsampling_mode_ca_context, upsampling_mode)
        
        self.start_from_noise = start_from_noise
        self.concat_context = concat_context
        self.ca_context = ca_context

        # forward diffusion of image
        self.noise_image = noising_step > 0
        self.noising_step = noising_step
        if self.start_from_noise and self.noise_image:
            raise ValueError("Cannot use noising step with start_from_noise=True")
        if self.noising_step > 0:
            if self.noising_step > 1 and isinstance(self.noising_step, int):
                self.diffusion = ForwardDiffusion()
            else:
                raise ValueError("Invalid noising step")
        else:
            self.diffusion = None

        # first stage encoding
        self.scale_factor = scale_factor
        if exists(first_stage_cfg):
            self.first_stage = instantiate_from_config(first_stage_cfg)
            freeze(self.first_stage)
            self.first_stage.eval()
            if self.scale_factor == 1.0:
                warnings.warn("Using first stage with scale_factor=1.0")
        else:
            if self.scale_factor != 1.0:
                raise ValueError("Cannot use scale_factor with identity first stage")
            self.first_stage = None
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.log_grad_norm = log_grad_norm

        self.vis_samples = None
        self.metric_tracker = instantiate_from_config(metric_tracker_cfg) if exists(metric_tracker_cfg) else None

        self.n_images_to_vis = n_images_to_vis
        self.val_epochs = 0

        self.save_hyperparameters()

        # flag to make sure the signal is not handled at an incorrect state, e.g. during weights update
        self.stop_training = False

    def stop_training_method(self):
        # dummy function to be compatible
        pass

    def upsample_latent(self, lres_z: Tensor, z_size: int, lres_ims: Tensor = None, im_size: int = None, mode: str = "bilinear", **kwargs):
        """
        已移除：超分辨率相关上采样方法。Trainer 现在专注于图像还原任务。
        """
        raise NotImplementedError("upsample_latent has been removed in restore-only mode. For upsampling features, restore the original branch or use a separate module.")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        out = dict(optimizer=opt)
        if exists(self.lr_scheduler_cfg):
            sch = load_partial_from_config(self.lr_scheduler_cfg)
            sch = sch(optimizer=opt)
            out["lr_scheduler"] = sch
        return out

    def forward(self, x_target: Tensor, x_source: Tensor, **kwargs):
        return self.model.training_losses(x1=x_target, x0=x_source, **kwargs)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if not exists(self.first_stage):
            return x
        x = self.first_stage.encode(x)
        if not isinstance(x, torch.Tensor): # hack for posterior of original VAE
            x = x.mode()
        return x * self.scale_factor

    @torch.no_grad()
    def decode_first_stage(self, z):
        if not exists(self.first_stage):
            return z
        return self.first_stage.decode(z / self.scale_factor)
    
    def extract_from_batch(self, batch):
        """
        Takes batch and extracts high-res and low-res images and latent codes.

        此函数专注于图像还原任务（Image Restoration）：
        - 输入: batch 必须包含 `image`（目标/clean）和 `image_degraded`（输入/degraded）
        - 返回:
            hres_ims: high-res images (target/clean images)
            hres_z: high-res latent codes (if identity first stage, this is hres_ims)
            lres_ims: low-res images (input/degraded images)
            lres_z: low-res latent codes (if identity first stage, this is lres_ims)
        """
        # 强制使用图像还原数据格式
        if "image_degraded" not in batch:
            raise KeyError("extract_from_batch: batch must contain 'image_degraded' for image restoration task")

        # 图像还原任务：受损图像 -> 无损图像
        lres_ims = batch["image_degraded"]  # 受损图像（输入）
        hres_ims = batch["image"]  # 无损图像（目标）

        # check if we have a pre-computed latent code for the high-res image
        if "latent" in batch:
            hres_z = batch["latent"]
            hres_z = hres_z * self.scale_factor
        else:
            hres_z = self.encode_first_stage(hres_ims)

        # check if we have a pre-computed latent code for the low-res image
        if "latent_lowres" in batch:
            lres_z = batch["latent_lowres"]
            lres_z = lres_z * self.scale_factor
        else:
            # encode to latent space (if no first stage, this is identity)
            lres_z = self.encode_first_stage(lres_ims)

        return hres_ims.float(), hres_z.float(), lres_ims.float(), lres_z.float()
        
    def training_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """ extract high-res and low-res images from batch

        Supports multiple training dataloaders. When using multiple
        DataLoaders with Lightning (e.g. passing a list from
        train_dataloader), Lightning will call training_step with
        an additional dataloader_idx parameter so you can distinguish
        the source. The training logic is identical for both datasets
        by default.
        """
        hres_ims, hres_z, lres_ims, lres_z = self.extract_from_batch(batch)

        """ context & conditioning information """
        x_source, context, context_ca = self.get_source_and_context(
            lres_z=lres_z,
            z_size=hres_z.shape[-1],
            lres_ims=lres_ims,
            im_size=hres_ims.shape[-1]
        )

        """ loss """
        loss = self.forward(x_target=hres_z, x_source=x_source, context=context, context_ca=context_ca)
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=x_source.shape[0])
        
        # 每100步打印一次训练进度（包含来自哪个 dataloader）
        if self.global_step % 100 == 0:
            self.print(f"Step {self.global_step} (loader={dataloader_idx}): Loss = {loss.item():.6f}")

        """ misc """
        self.ema_model.update()
        if exists(self.lr_scheduler_cfg):
            self.lr_schedulers().step()
        if self.stop_training:
            self.stop_training_method()
        if self.log_grad_norm:
            grad_norm = get_grad_norm(self.model)
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False)

        return loss

    def get_source_and_context(self, lres_z: Tensor, z_size: int, lres_ims: Tensor = None, im_size: int = None):
        """
        Args:
            lres_z: low-res latent code (in low-res space)
            z_size: size of the high-res latent code (for restore tasks this typically equals lres_z.shape[-1])
            lres_ims: low-res images in pixel space
            im_size: size of the high-res images in pixel space
        Returns:
            x_source: x0 for the flow matching model
            context: context for the flow matching model
        """
        # 对于图像还原，通常输入输出分辨率相同，因此不需要复杂的上采样逻辑
        if z_size == lres_z.shape[-1] or self.upsampling_mode == "identity":
            lres_z_hr = lres_z
        else:
            # 退化为在潜在空间插值上采样（保留简单插值以兼容部分配置）
            try:
                lres_z_hr = nn.functional.interpolate(lres_z, size=z_size, mode=self.upsampling_mode)
            except Exception:
                # fallback to bilinear
                lres_z_hr = nn.functional.interpolate(lres_z, size=z_size, mode="bilinear")

        # define x0
        if self.start_from_noise:
            x_source = torch.randn_like(lres_z_hr)
        else:
            x_source = lres_z_hr

        # noise the start
        if self.noise_image:
            x_source = self.diffusion.q_sample(x_start=x_source, t=self.noising_step)
        
        # define context (concatenated with image latent codes)
        if self.concat_context or self.start_from_noise:
            # context 默认为上采样后的 latent（或原始 latent）
            context = lres_z_hr
        else:
            context = None

        # cross-attention context
        if self.ca_context:
            context_ca = lres_z_hr
        else:
            context_ca = None

        return x_source, context, context_ca

    def predict_restored_img(self, lres_z: Tensor, sample_kwargs=None):
        """
        在相同分辨率潜在空间中执行图像还原推理（用于 validation / 推理）。
        Args:
            lres_z: low-res/degraded latent codes (B, C, H, W)
            sample_kwargs: dict, 传递给 generate 的参数
        Returns:
            restored_img: 解码后的图像（B, C, H_pixel, W_pixel）
        """
        # z_size 采用输入latent的空间尺寸
        z_size = lres_z.shape[-1]
        x_source, context, context_ca = self.get_source_and_context(lres_z=lres_z, z_size=z_size, lres_ims=None, im_size=None)

        if not exists(sample_kwargs):
            sample_kwargs = dict(num_steps=40, method="rk4")

        fn = self.ema_model.model.generate if self.use_ema_for_sampling else self.model.generate
        restored_z = fn(x=x_source, context=context, context_ca=context_ca, sample_kwargs=sample_kwargs)
        restored_img = self.decode_first_stage(restored_z)
        return restored_img

    @torch.no_grad()
    def predict_restored_img_padded(self, ims: Tensor, sample_kwargs=None, multiple: int = 8):
        """Convenience wrapper for inference on arbitrary-sized pixel images.

        Steps:
        - pads the input image so H and W are divisible by `multiple`
        - encodes with first_stage (if present)
        - calls `predict_restored_img` on the latent
        - decodes and removes padding

        Args:
            ims: pixel images tensor (B, C, H, W) in range [-1, 1]
            sample_kwargs: passed to predict_restored_img
            multiple: spatial multiple to pad to (default 8)
        Returns:
            restored images tensor cropped to original size
        """
        # local import to avoid circular imports at module import time
        from cfm.helpers import pad_to_multiple, unpad

        # pad images to required multiple
        ims_padded, pads = pad_to_multiple(ims, multiple=multiple, mode='reflect')

        # encode to latent space (if no first stage, encode_first_stage returns input)
        lres_z_padded = self.encode_first_stage(ims_padded)

        # run existing latent-space inference
        restored_z_padded = self.predict_restored_img(lres_z_padded, sample_kwargs=sample_kwargs)

        # decode to pixel space (if first stage exists, decode_first_stage will be used)
        restored_img_padded = self.decode_first_stage(restored_z_padded)

        # unpad back to original image size
        restored_img = unpad(restored_img_padded, pads)
        return restored_img

    def validation_step(self, batch, batch_idx):
        hr_ims, hr_z, lr_ims, lr_z = self.extract_from_batch(batch)
        # 图像还原：直接在相同潜在分辨率上进行还原
        hr_pred = self.predict_restored_img(lres_z=lr_z, sample_kwargs=dict(num_steps=40, method="rk4"))

        # track metrics
        if exists(self.metric_tracker):
            self.metric_tracker(hr_ims, hr_pred)

        if self.stop_training:
            self.stop_training_method()
        
        # store samples for visualization
        if self.vis_samples is None:
            self.vis_samples = {'hr': hr_ims, 'lr': lr_ims, 'pred': hr_pred}
        elif self.vis_samples['hr'].shape[0] < self.n_images_to_vis:
            self.vis_samples['hr'] = torch.cat([self.vis_samples['hr'], hr_ims], dim=0)
            self.vis_samples['lr'] = torch.cat([self.vis_samples['lr'], lr_ims], dim=0)
            self.vis_samples['pred'] = torch.cat([self.vis_samples['pred'], hr_pred], dim=0)

    def on_validation_epoch_end(self):
        # log low-res images, high-res images, and up-sampled images
        out_img = hres_lres_pred_grid(self.vis_samples['hr'], self.vis_samples['lr'], self.vis_samples['pred'])
        self.log_image(out_img, "val")
        self.vis_samples = None

        # compute metrics
        if exists(self.metric_tracker):
            metrics = self.metric_tracker.aggregate()
            # 格式化输出指标
            metric_str = ", ".join([f"{k.upper()}={v:.4f}" for k, v in metrics.items()])
            for k, v in metrics.items():
                self.log(f"val/{k}", v, sync_dist=True)
            self.metric_tracker.reset()
        else:
            metric_str = "No metrics"
        
        self.val_epochs += 1
        self.print(f"\n{'='*60}")
        self.print(f"Validation Epoch {self.val_epochs} | Step {self.global_step}")
        self.print(f"Metrics: {metric_str}")
        self.print(f"{'='*60}\n")

        torch.cuda.empty_cache()

    def log_image(self, img, name):
        """
        Args:
            ims: torch.Tensor or np.ndarray of shape (h, w, c) in range [0, 255]
            name: str
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if isinstance(self.logger, WandbLogger):
            img = Image.fromarray(img)
            img = wandb.Image(img)
            self.logger.experiment.log({f"{name}/samples": img}, step=self.global_step)
        else:
            img = einops.rearrange(img, "h w c -> c h w")
            self.logger.experiment.add_image(f"{name}/samples", img, global_step=self.global_step)


def get_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

import os
import argparse
import datetime

import torch
from torch import set_float32_matmul_precision
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from fmboost.helpers import instantiate_from_config
"""
python validate_only.py --config configs/flow400_64-128/unet-base_psu.yaml \
    --ckpt logs/<train_exp>/checkpoints/last.ckpt \
    --name myexp --devices 1
"""

def parse_args():
    parser = argparse.ArgumentParser("FMBoost validate-only")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint to evaluate")
    parser.add_argument("--name", type=str, default="eval", help="Experiment name prefix")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use (set -1 for all)")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Precision for evaluation (e.g., 16-mixed, bf16-mixed, 32-true)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override val/test batch size to save memory")
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def build_loggers(log_dir, cfg):
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name="",
        version="",
        log_graph=False,
        default_hp_metric=False,
    )
    csv_logger = CSVLogger(
        log_dir,
        name="",
        version="",
        prefix="",
        flush_logs_every_n_steps=100,
    )
    csv_logger.log_hyperparams(OmegaConf.to_container(cfg))
    return [tb_logger, csv_logger]


def select_loader(data_module):
    if hasattr(data_module, "test_dataloader"):
        loader = data_module.test_dataloader()
        if loader is not None and loader != []:
            return loader, "test"
    loader = data_module.val_dataloader()
    return loader, "val"


def main():
    args = parse_args()
    seed_everything(args.seed)

    # Enable Tensor Core friendly matmul precision for speed on modern GPUs.
    set_float32_matmul_precision("medium")

    cfg = OmegaConf.load(args.config)

    # log dir: logs/test/<name>_<timestamp>
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_name = f"{args.name}_{now}"
    base_log_dir = os.path.join("logs", "test")
    os.makedirs(base_log_dir, exist_ok=True)
    log_dir = os.path.join(base_log_dir, exp_name)

    loggers = build_loggers(log_dir, cfg)

    data = instantiate_from_config(cfg.data)
    module = instantiate_from_config(cfg.model)

    # allow overriding batch size at evaluation time to reduce memory
    if args.batch_size is not None:
        if hasattr(data, "test_batch_size"):
            data.test_batch_size = args.batch_size
        if hasattr(data, "val_batch_size"):
            data.val_batch_size = args.batch_size
        if hasattr(data, "batch_size"):
            data.batch_size = args.batch_size

    # propagate split name to module for metric naming
    if hasattr(module, "log_prefix"):
        module.log_prefix = "test"

    # ensure datasets are built
    if hasattr(data, "setup"):
        data.setup("test")

    # choose device setup similar to train
    if torch.cuda.is_available():
        gpu_kwargs = {"accelerator": "gpu"}
        if args.devices == -1:
            gpu_kwargs["devices"] = torch.cuda.device_count()
        else:
            gpu_kwargs["devices"] = args.devices
    else:
        gpu_kwargs = {"accelerator": "cpu"}

    trainer = Trainer(logger=loggers, precision=args.precision, **gpu_kwargs)

    dataloader, split = select_loader(data)
    if dataloader is None or dataloader == []:
        raise RuntimeError(f"No {split} dataloader available for evaluation")

    print("-" * 40)
    print(f"Config        : {args.config}")
    print(f"Checkpoint    : {args.ckpt}")
    print(f"Log dir       : {log_dir}")
    print(f"Split         : {split}")
    print(f"Precision     : {args.precision}")
    if args.batch_size is not None:
        print(f"Eval batch    : {args.batch_size}")
    if split == "test":
        print("Using TEST split (root/test/...)")
    else:
        print("Using VAL split (root/val/...) — test split not found")
    print("-" * 40)

    print("Evaluating on test set...")
    trainer.test(model=module, dataloaders=dataloader, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()

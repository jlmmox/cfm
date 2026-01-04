import argparse
import os
import random
import shutil
from pathlib import Path


def collect_pairs(deg_dir: Path, gt_dir: Path):
    def map_stems(folder: Path):
        m = {}
        for p in folder.iterdir():
            if p.is_file():
                m[p.stem] = p
        return m

    dm = map_stems(deg_dir)
    gm = map_stems(gt_dir)
    keys = sorted(set(dm.keys()) & set(gm.keys()))
    return [(dm[k], gm[k]) for k in keys]


def main():
    parser = argparse.ArgumentParser(
        description="Split a portion of train/{input,target} into val/{input,target} by filename stem matching."
    )
    parser.add_argument("--root", required=True, help="Dataset root that contains train/ and optional val/")
    parser.add_argument("--degraded_dir", default="input", help="Degraded subfolder name under each split")
    parser.add_argument("--clean_dir", default="target", help="Clean/GT subfolder name under each split")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Portion to move/copy from train to val")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--copy", action="store_true", help="Copy instead of move (default: move)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files in val if present")
    args = parser.parse_args()

    # Expand '~' to user home to avoid literal tilde issues
    root = Path(os.path.expanduser(args.root))
    train_deg = root / "train" / args.degraded_dir
    train_gt = root / "train" / args.clean_dir
    val_deg = root / "val" / args.degraded_dir
    val_gt = root / "val" / args.clean_dir

    if not train_deg.is_dir() or not train_gt.is_dir():
        raise FileNotFoundError(f"Expected train/{args.degraded_dir} and train/{args.clean_dir} under {root}")

    val_deg.mkdir(parents=True, exist_ok=True)
    val_gt.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(train_deg, train_gt)
    if not pairs:
        raise RuntimeError("No paired files found by matching filename stems between degraded and clean.")

    n_total = len(pairs)
    n_val = max(1, int(n_total * args.val_ratio))
    random.seed(args.seed)
    random.shuffle(pairs)
    chosen = pairs[:n_val]

    op = shutil.copy2 if args.copy else shutil.move
    moved = 0
    skipped = 0
    for d_src, g_src in chosen:
        d_dst = val_deg / d_src.name
        g_dst = val_gt / g_src.name
        if not args.force and (d_dst.exists() or g_dst.exists()):
            skipped += 1
            continue
        if args.force:
            if d_dst.exists():
                d_dst.unlink()
            if g_dst.exists():
                g_dst.unlink()
        op(str(d_src), str(d_dst))
        op(str(g_src), str(g_dst))
        moved += 1

    print("-" * 40)
    print(f"Root              : {root}")
    print(f"Train pairs found : {n_total}")
    print(f"Val ratio         : {args.val_ratio}")
    print(f"Selected for val  : {n_val}")
    print(f"Action            : {'COPY' if args.copy else 'MOVE'}")
    print(f"Moved/Copied      : {moved}")
    print(f"Skipped (exists)  : {skipped}")
    print(f"Val degraded dir  : {val_deg}")
    print(f"Val clean dir     : {val_gt}")
    print("-" * 40)


if __name__ == "__main__":
    main()

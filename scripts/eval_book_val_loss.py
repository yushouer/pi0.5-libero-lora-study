import argparse
import hashlib
import json
import os
from pathlib import Path

# 保持之前解决段错误的导入顺序
import openpi.training.checkpoints
import torch

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.shared import nnx_utils
from openpi.training import config, data_loader


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    root = Path("results/book_task09")
    val_file = (root / "val_episodes.json").resolve()
    ckpt = Path(
        f"checkpoints/pi05_libero_book_lora/"
        f"book_6k_v1/{args.step}"
    )

    assert val_file.is_file(), val_file
    assert (ckpt / "params").is_dir(), ckpt

    # 只读取验证集，不读取训练集
    os.environ["OPENPI_EPISODES_FILE"] = str(val_file)

    cfg = config.get_config("pi05_libero_book_lora")

    # 确认验证时的统计量与 checkpoint 一致
    asset = Path("physical-intelligence/libero/norm_stats.json")
    local_stats = Path("assets/pi05_libero_book_lora") / asset
    ckpt_stats = ckpt / "assets" / asset

    assert local_stats.is_file() and ckpt_stats.is_file()
    assert sha256(local_stats) == sha256(ckpt_stats), (
        "Normalization stats differ from checkpoint!"
    )

    # 验证集不能重新拟合归一化参数
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert data_cfg.norm_stats is not None

    ds = data_loader.create_torch_dataset(
        data_cfg,
        cfg.model.action_horizon,
        cfg.model,
    )

    assert len(ds) == 1222, f"Expected 1222, got {len(ds)}"

    allowed = set(json.loads(val_file.read_text()))
    for idx in (0, 429, len(ds) - 1):
        assert int(ds[idx]["episode_index"]) in allowed

    n = len(ds) if args.limit <= 0 else min(args.limit, len(ds))

    loader = data_loader.create_data_loader(
        cfg,
        shuffle=False,
        num_batches=n,
    )

    print(f"Checkpoint: {ckpt}")
    print(f"Validation samples: {n}")
    print("Loading model...", flush=True)

    params = model_lib.restore_params(
        ckpt / "params",
        dtype=jnp.bfloat16,
    )
    model = cfg.model.load(params)
    model.eval()

    # 官方用于模型推理的低显存 JIT 封装
    loss_fn = nnx_utils.module_jit(model.compute_loss)

    base_rng = jax.random.key(7)
    total_loss = 0.0
    count = 0

    for i, (obs, actions) in enumerate(loader):
        # 三个 checkpoint 使用相同的噪声和时间采样
        rng = jax.random.fold_in(base_rng, i)

        losses = loss_fn(rng, obs, actions)
        value = float(jnp.mean(losses))

        if not np.isfinite(value):
            raise RuntimeError(f"Non-finite loss at sample {i}")

        total_loss += value
        count += 1

        if count % 100 == 0 or count == n:
            print(
                f"Evaluated {count}/{n}, "
                f"mean_loss={total_loss/count:.6f}",
                flush=True,
            )

        if count >= n:
            break

    assert count == n, f"Expected {n}, evaluated {count}"

    result = {
        "checkpoint_step": args.step,
        "split": "val",
        "num_samples": count,
        "seed": 7,
        "train_mode": False,
        "mean_flow_matching_loss": total_loss / count,
    }

    out = root / "validation"
    out.mkdir(parents=True, exist_ok=True)

    path = out / f"step{args.step}_val_{n}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")

    print("\nValidation result:")
    print(json.dumps(result, indent=2))
    print("Saved:", path)


if __name__ == "__main__":
    main()

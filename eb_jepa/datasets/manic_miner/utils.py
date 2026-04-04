"""Data loading utility for Manic Miner datasets."""

import torch
from .dataset import ManicMinerDataset, ManicMinerDataConfig


def init_manic_miner_data(cfg_data: dict):
    """Initialise train/val loaders from config dict.

    Returns (train_loader, val_loader, config).
    """
    cfg = ManicMinerDataConfig(
        data_dirs=cfg_data.get("data_dirs", ["datasets/cma-ea/manic-miner"]),
        batch_size=cfg_data.get("batch_size", 64),
        num_workers=cfg_data.get("num_workers", 4),
        pin_mem=cfg_data.get("pin_mem", True),
        persistent_workers=cfg_data.get("persistent_workers", True),
        seq_len=cfg_data.get("seq_len", 16),
        frameskip=cfg_data.get("frameskip", 1),
        max_shards=cfg_data.get("max_shards", 0),
        img_size=cfg_data.get("img_size", 64),
    )

    dset = ManicMinerDataset(cfg)
    cfg.size = len(dset)

    # 90/10 train/val split
    n_val = max(1, len(dset) // 10)
    n_train = len(dset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    cfg.val_size = len(val_set)

    loader_kw = dict(
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_mem,
        drop_last=True,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True, **loader_kw,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=min(cfg.batch_size, 8), shuffle=False, **loader_kw,
    )

    return train_loader, val_loader, cfg

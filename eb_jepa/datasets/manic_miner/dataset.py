"""
Manic Miner (ZX Spectrum) trajectory dataset.

Reads msgpack shard files produced by ea_train.py.  Each shard is a list of
dicts with keys:
    obs.screen       – 49152 floats (192*256, bit-unpacked bitmap)
    obs.attr_grid    – 768 floats (24*32 attribute bytes / 255)
    obs.position_vec – [col/31, row/15, room/20]
    action           – int  (0..4: noop,left,right,jump,jump_right)
    reward           – float
    terminated       – bool
    truncated        – bool
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import msgpack
import numpy as np
import torch
from torch.utils.data import Dataset

# ZX Spectrum display: 192 rows x 256 cols (monochrome bitmap)
SCREEN_H, SCREEN_W = 192, 256
N_ACTIONS = 5


@dataclass
class ManicMinerDataConfig:
    data_dirs: List[str] = field(default_factory=lambda: ["datasets/cma-ea/manic-miner"])
    batch_size: int = 64
    num_workers: int = 4
    pin_mem: bool = True
    persistent_workers: bool = True
    seq_len: int = 16       # number of frames per training slice
    frameskip: int = 1
    max_shards: int = 0     # 0 = load all
    img_size: int = 64      # resized image side (H and W after downscale)
    size: int = 0           # filled in after loading
    val_size: int = 0


class ManicMinerDataset(Dataset):
    """
    Loads trajectory shards and exposes them as (obs, actions, positions)
    sequences suitable for eb_jepa training.

    Returns per __getitem__:
        obs     – Tensor [T, 1, img_size, img_size]
        actions – Tensor [T, N_ACTIONS]   one-hot
        locs    – Tensor [T, 3]           (col, row, room) normalised
    """

    def __init__(self, cfg: ManicMinerDataConfig):
        self.cfg = cfg
        self.img_size = cfg.img_size
        self.seq_len = cfg.seq_len
        self.action_dim = N_ACTIONS
        self.proprio_dim = 3
        self.state_dim = 3

        # Load all shards into memory
        self.trajectories: List[list] = []
        self._load_shards(cfg.data_dirs, cfg.max_shards)

        # Build slice index: (traj_idx, start)
        self.slices = []
        for ti, traj in enumerate(self.trajectories):
            T = len(traj)
            for start in range(T - cfg.seq_len + 1):
                self.slices.append((ti, start))
        np.random.shuffle(self.slices)

    def _load_shards(self, data_dirs: List[str], max_shards: int):
        paths = []
        for d in data_dirs:
            p = Path(d)
            paths.extend(sorted(p.glob("shard_*.msgpack")))
        if max_shards > 0:
            paths = paths[:max_shards]
        for sp in paths:
            with open(sp, "rb") as f:
                steps = msgpack.unpack(f, raw=False)
            if len(steps) >= self.seq_len:
                self.trajectories.append(steps)

    @staticmethod
    def _screen_to_image(screen_flat: list, img_size: int) -> np.ndarray:
        """Convert 49152-float flat bitmap → [1, img_size, img_size] float32."""
        arr = np.array(screen_flat, dtype=np.float32).reshape(SCREEN_H, SCREEN_W)
        # Downsample via block-mean to img_size x img_size
        bh, bw = SCREEN_H // img_size, SCREEN_W // img_size
        # Use reshape + mean trick (only works when dims divide evenly)
        # Pad / truncate if needed
        h_crop = img_size * bh
        w_crop = img_size * bw
        arr = arr[:h_crop, :w_crop]
        arr = arr.reshape(img_size, bh, img_size, bw).mean(axis=(1, 3))
        return arr[np.newaxis, :, :]  # [1, H, W]

    @staticmethod
    def _action_onehot(action: int) -> np.ndarray:
        v = np.zeros(N_ACTIONS, dtype=np.float32)
        v[int(action) % N_ACTIONS] = 1.0
        return v

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        ti, start = self.slices[idx]
        steps = self.trajectories[ti][start : start + self.seq_len]

        obs_list, act_list, loc_list = [], [], []
        for s in steps:
            obs_list.append(self._screen_to_image(s["obs"]["screen"], self.img_size))
            act_list.append(self._action_onehot(s["action"]))
            loc_list.append(np.array(s["obs"]["position_vec"], dtype=np.float32))

        obs     = torch.from_numpy(np.stack(obs_list))      # [T, 1, H, W]
        actions = torch.from_numpy(np.stack(act_list))       # [T, 5]
        locs    = torch.from_numpy(np.stack(loc_list))       # [T, 3]

        # eb_jepa expects obs as [C, T, H, W]
        obs = obs.permute(1, 0, 2, 3)  # [1, T, H, W]

        # wall_x / door_y dummies to match WallSample format
        dummy = torch.tensor([0.0])
        return obs, actions, locs, dummy, dummy

    @property
    def normalizer(self):
        """Minimal normalizer that satisfies the training loop interface."""
        return _IdentityNormalizer()


class _IdentityNormalizer:
    """Pass-through normalizer for Manic Miner (positions already in [0,1])."""
    def normalize(self, x):
        return x
    def unnormalize(self, x):
        return x
    def unnormalize_mse(self, loss):
        return loss

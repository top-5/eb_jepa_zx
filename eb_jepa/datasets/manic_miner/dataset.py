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

Features:
    - Horizontal flip augmentation (p=0.5): mirrors the frame AND swaps
      left↔right / jump_right↔jump_left actions.  Teaches JEPA that jump
      physics are symmetric.
    - Optional 2-channel mode (use_attr_grid=True): stacks the monochrome
      bitmap with the 24×32 attribute grid (upscaled to img_size) so the
      encoder can see platform structure directly.
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
# Attribute grid: 24 rows x 32 cols
ATTR_H, ATTR_W = 24, 32
N_ACTIONS = 6

# Action indices: NOOP=0, LEFT=1, RIGHT=2, JUMP=3, LEFT+JUMP=4, RIGHT+JUMP=5
A_NOOP, A_LEFT, A_RIGHT, A_JUMP, A_LEFT_JUMP, A_RIGHT_JUMP = range(N_ACTIONS)

# Mirror map for horizontal flip augmentation
FLIP_ACTION = {
    A_NOOP:       A_NOOP,
    A_LEFT:       A_RIGHT,
    A_RIGHT:      A_LEFT,
    A_JUMP:       A_JUMP,
    A_LEFT_JUMP:  A_RIGHT_JUMP,
    A_RIGHT_JUMP: A_LEFT_JUMP,
}


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
    hflip_prob: float = 0.5     # horizontal flip augmentation probability
    use_attr_grid: bool = False  # if True, stack attr grid as 2nd channel (dobs=2)


class ManicMinerDataset(Dataset):
    """
    Loads trajectory shards and exposes them as (obs, actions, positions)
    sequences suitable for eb_jepa training.

    Returns per __getitem__:
        obs     – Tensor [C, T, img_size, img_size]  C=1 or 2
        actions – Tensor [T, N_ACTIONS]   one-hot
        locs    – Tensor [T, 3]           (col, row, room) normalised
        dummy, dummy – zeros (WallSample compat)
    """

    def __init__(self, cfg: ManicMinerDataConfig):
        self.cfg = cfg
        self.img_size = cfg.img_size
        self.seq_len = cfg.seq_len
        self.hflip_prob = cfg.hflip_prob
        self.use_attr_grid = cfg.use_attr_grid
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
        """Convert 49152-float flat bitmap → [img_size, img_size] float32."""
        arr = np.array(screen_flat, dtype=np.float32).reshape(SCREEN_H, SCREEN_W)
        bh, bw = SCREEN_H // img_size, SCREEN_W // img_size
        h_crop = img_size * bh
        w_crop = img_size * bw
        arr = arr[:h_crop, :w_crop]
        arr = arr.reshape(img_size, bh, img_size, bw).mean(axis=(1, 3))
        return arr  # [H, W]

    @staticmethod
    def _attr_to_image(attr_flat: list, img_size: int) -> np.ndarray:
        """Convert 768-float attr grid → [img_size, img_size] float32.

        Nearest-neighbor upscale from 24×32 to img_size×img_size.
        """
        arr = np.array(attr_flat, dtype=np.float32).reshape(ATTR_H, ATTR_W)
        # Nearest-neighbor upscale via repeat
        row_scale = img_size // ATTR_H  # 64/24 ≈ 2.67 — doesn't divide evenly
        col_scale = img_size // ATTR_W  # 64/32 = 2
        # Use np.kron for exact upscale, then crop/resize
        # For simplicity, use repeat + slice
        up = np.repeat(np.repeat(arr, row_scale + 1, axis=0), col_scale, axis=1)
        return up[:img_size, :img_size]  # [H, W]

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

        do_flip = self.hflip_prob > 0 and np.random.random() < self.hflip_prob

        obs_list, act_list, loc_list = [], [], []
        for s in steps:
            bitmap = self._screen_to_image(s["obs"]["screen"], self.img_size)

            if self.use_attr_grid:
                attr = self._attr_to_image(s["obs"]["attr_grid"], self.img_size)
                frame = np.stack([bitmap, attr], axis=0)  # [2, H, W]
            else:
                frame = bitmap[np.newaxis, :, :]  # [1, H, W]

            if do_flip:
                frame = frame[:, :, ::-1].copy()  # flip W axis

            obs_list.append(frame)

            action = int(s["action"])
            if do_flip:
                action = FLIP_ACTION.get(action, action)
            act_list.append(self._action_onehot(action))

            loc = np.array(s["obs"]["position_vec"], dtype=np.float32)
            if do_flip:
                loc[0] = 1.0 - loc[0]  # mirror col coordinate
            loc_list.append(loc)

        obs     = torch.from_numpy(np.stack(obs_list))      # [T, C, H, W]
        actions = torch.from_numpy(np.stack(act_list))       # [T, 5]
        locs    = torch.from_numpy(np.stack(loc_list))       # [T, 3]

        # eb_jepa expects obs as [C, T, H, W]
        obs = obs.permute(1, 0, 2, 3)  # [C, T, H, W]

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

"""
Train an action-conditioned Video JEPA on Manic Miner trajectories.

Adapted from examples/ac_video_jepa/main.py for:
- 1-channel monochrome input (dobs=1)
- 5 discrete actions (one-hot encoded)
- 3D position vector (col, row, room)
"""

import os
from pathlib import Path
from time import time

import fire
import torch
import torch.nn as nn
import wandb
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from eb_jepa.architectures import (
    ImpalaEncoder,
    InverseDynamicsModel,
    Projector,
    RNNPredictor,
)
from eb_jepa.datasets.utils import init_data
from eb_jepa.jepa import JEPA, JEPAProbe
from eb_jepa.logging import get_logger
from eb_jepa.losses import SquareLossSeq, VC_IDM_Sim_Regularizer
from eb_jepa.schedulers import CosineWithWarmup
from eb_jepa.state_decoder import MLPXYHead
from eb_jepa.training_utils import (
    get_default_dev_name,
    get_exp_name,
    get_unified_experiment_dir,
    load_checkpoint,
    load_config,
    log_config,
    log_data_info,
    log_epoch,
    log_model_info,
    save_checkpoint,
    setup_device,
    setup_seed,
    setup_wandb,
)

logger = get_logger(__name__)

ACTION_DIM = 6  # noop, left, right, jump, left+jump, right+jump


def run(
    fname: str = "examples/manic_miner/cfgs/train.yaml",
    cfg=None,
    folder=None,
    **overrides,
):
    """Train an action-conditioned Video JEPA on Manic Miner data."""
    if cfg is None:
        cfg = load_config(fname, overrides if overrides else None)

    # Experiment directory
    if folder is None:
        if cfg.meta.get("model_folder"):
            folder = Path(cfg.meta.model_folder)
            exp_name = folder.name.rsplit("_seed", 1)[0]
        else:
            sweep_name = get_default_dev_name()
            exp_name = get_exp_name("manic_miner", cfg)
            folder = get_unified_experiment_dir(
                example_name="manic_miner",
                sweep_name=sweep_name,
                exp_name=exp_name,
                seed=cfg.meta.seed,
            )
    else:
        folder = Path(folder)
        exp_name = folder.name.rsplit("_seed", 1)[0]

    os.makedirs(folder, exist_ok=True)

    # -- DATA
    loader, val_loader, data_config = init_data(
        env_name=cfg.data.env_name, cfg_data=dict(cfg.data)
    )

    # -- SETUP
    setup_device("auto")
    setup_seed(cfg.meta.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- WANDB
    wandb_run = setup_wandb(
        project="eb_jepa_zx",
        config={
            "example": "manic_miner",
            **OmegaConf.to_container(cfg, resolve=True),
        },
        run_dir=folder,
        run_name=exp_name,
        tags=[f"seed_{cfg.meta.seed}", "manic_miner"],
        group=cfg.logging.get("wandb_group"),
        enabled=cfg.logging.get("log_wandb", False),
        sweep_id=cfg.logging.get("wandb_sweep_id"),
    )

    log_data_info(
        cfg.data.env_name,
        len(loader),
        data_config.batch_size,
        train_samples=data_config.size,
        val_samples=data_config.val_size,
    )

    # Mixed precision
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map.get(cfg.training.get("dtype", "float16").lower(), torch.float16) if hasattr(cfg, "training") else torch.float16
    use_amp = cfg.training.get("use_amp", True) if hasattr(cfg, "training") else True
    scaler = GradScaler(device.type, enabled=use_amp)
    logger.info(f"Using AMP with {dtype=}" if use_amp else "AMP disabled")

    # -- SAVE CONFIG
    latest_ckpt_path = folder / "latest.pth.tar"
    steps_per_epoch = max(1, data_config.size // data_config.batch_size)
    total_steps = cfg.optim.epochs * steps_per_epoch
    config_path = folder / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, config_path)
    logger.info(f"Saved config to {config_path}")

    action_dim = cfg.model.get("action_dim", ACTION_DIM)

    # -- MODEL
    test_input = torch.rand(
        1, cfg.model.dobs, 1, data_config.img_size, data_config.img_size
    )
    encoder = ImpalaEncoder(
        width=1,
        stack_sizes=(16, cfg.model.henc, cfg.model.dstc),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=cfg.model.dobs,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(cfg.model.dobs, data_config.img_size, data_config.img_size),
    )
    test_output = encoder(test_input)
    _, f, _, h, w = test_output.shape

    predictor = RNNPredictor(
        hidden_size=encoder.mlp_output_dim,
        final_ln=encoder.final_ln,
        action_dim=action_dim,
    )

    aencoder = nn.Identity()

    if cfg.model.regularizer.use_proj:
        projector = Projector(
            f"{encoder.mlp_output_dim}-{encoder.mlp_output_dim*4}-{encoder.mlp_output_dim*4}"
        )
    else:
        projector = None

    logger.info(f"Encoder output: {tuple(test_output.shape)}")

    idm = InverseDynamicsModel(
        state_dim=h * w * (projector.out_dim if cfg.model.regularizer.idm_after_proj else f),
        hidden_dim=256,
        action_dim=action_dim,
    ).to(device)

    regularizer = VC_IDM_Sim_Regularizer(
        cov_coeff=cfg.model.regularizer.cov_coeff,
        std_coeff=cfg.model.regularizer.std_coeff,
        sim_coeff_t=cfg.model.regularizer.sim_coeff_t,
        idm_coeff=cfg.model.regularizer.get("idm_coeff", 0.1),
        idm=idm,
        first_t_only=cfg.model.regularizer.get("first_t_only"),
        projector=projector,
        spatial_as_samples=cfg.model.regularizer.spatial_as_samples,
        idm_after_proj=cfg.model.regularizer.idm_after_proj,
        sim_t_after_proj=cfg.model.regularizer.sim_t_after_proj,
    )

    ploss = SquareLossSeq()
    jepa = JEPA(encoder, aencoder, predictor, regularizer, ploss).to(device)

    encoder_params = sum(p.numel() for p in encoder.parameters())
    predictor_params = sum(p.numel() for p in predictor.parameters())
    log_model_info(jepa, {"encoder": encoder_params, "predictor": predictor_params})
    log_config(cfg)

    # -- PROBER (position prediction head)
    xy_head = MLPXYHead(
        input_shape=test_output.shape[1],
        normalizer=loader.dataset.dataset.normalizer
        if hasattr(loader.dataset, "dataset")
        else loader.dataset.normalizer,
    ).to(device)
    xy_prober = JEPAProbe(jepa=jepa, head=xy_head, hcost=nn.MSELoss())

    # -- OPTIMIZERS
    jepa_optimizer = AdamW(
        jepa.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.get("weight_decay", 1e-6),
    )
    jepa_scheduler = CosineWithWarmup(jepa_optimizer, total_steps, warmup_ratio=0.1)

    probe_optimizer = AdamW(xy_head.parameters(), lr=1e-3, weight_decay=1e-5)
    probe_scheduler = CosineWithWarmup(probe_optimizer, total_steps, warmup_ratio=0.1)

    # -- LOAD CHECKPOINT
    start_epoch = 0
    if cfg.meta.load_model:
        checkpoint_path = folder / cfg.meta.get("load_checkpoint", "latest.pth.tar")
        ckpt_info = load_checkpoint(
            checkpoint_path, jepa, jepa_optimizer, jepa_scheduler, device=device
        )
        start_epoch = ckpt_info.get("epoch", 0)
        if "xy_head_state_dict" in ckpt_info:
            xy_head.load_state_dict(ckpt_info["xy_head_state_dict"])

    # Compile
    if torch.cuda.is_available() and cfg.model.compile:
        logger.info("Compiling model with torch.compile")
        jepa = torch.compile(jepa)

    # -- TRAINING LOOP
    for epoch in range(start_epoch, cfg.optim.epochs):
        epoch_start = time()
        pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"Epoch {epoch}/{cfg.optim.epochs - 1}",
            disable=cfg.logging.get("tqdm_silent", False),
        )
        for idx, (x, a, loc, _, _) in pbar:
            itr_start = time()
            global_step = epoch * len(loader) + idx

            x = x.to(device)
            a = a.to(device)
            loc = loc.to(device)
            total_loss = torch.tensor(0.0, device=device)

            # JEPA loss
            jepa_optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp, dtype=dtype):
                _, (jepa_loss, regl, regl_unweight, regldict, pl) = jepa.unroll(
                    x, a,
                    nsteps=cfg.model.nsteps,
                    unroll_mode="autoregressive",
                    ctxt_window_time=1,
                    compute_loss=True,
                    return_all_steps=False,
                )
                total_loss += jepa_loss

            scaler.scale(jepa_loss).backward()
            if cfg.optim.get("grad_clip_enc") and cfg.optim.get("grad_clip_pred"):
                scaler.unscale_(jepa_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    jepa.encoder.parameters(), cfg.optim.grad_clip_enc
                )
                torch.nn.utils.clip_grad_norm_(
                    jepa.predictor.parameters(), cfg.optim.grad_clip_pred
                )
            scaler.step(jepa_optimizer)
            scaler.update()
            jepa_scheduler.step()

            # Probe loss
            probe_optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp, dtype=dtype):
                xy_loss = xy_prober(
                    observations=x[:, :, :1],          # [B, C, 1, H, W] first frame
                    targets=loc[:, :1, :2].permute(0, 2, 1),  # [B, 2, 1] col+row for first frame
                )
                normalizer = (
                    loader.dataset.dataset.normalizer
                    if hasattr(loader.dataset, "dataset")
                    else loader.dataset.normalizer
                )
                xy_loss = normalizer.unnormalize_mse(xy_loss)
                total_loss += xy_loss

            scaler.scale(xy_loss).backward()
            scaler.step(probe_optimizer)
            scaler.update()
            probe_scheduler.step()

            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "reg": f"{regl.item():.4f}",
                "pred": f"{pl.item():.4f}",
            })

            itr_time = time() - itr_start
            if global_step % cfg.logging.log_every == 0:
                log_data = {
                    "train/total_loss": total_loss.item(),
                    "train/reg_loss": regl.item(),
                    "train/reg_loss_unweight": regl_unweight.item(),
                    "train/pred_loss": pl.item(),
                    "train/probe_loss": xy_loss.item(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "itr_time": itr_time,
                    "optim/jepa_lr": jepa_optimizer.param_groups[0]["lr"],
                    "optim/probe_lr": probe_optimizer.param_groups[0]["lr"],
                }
                for loss_name, loss_value in regldict.items():
                    log_data[f"train/regl/{loss_name}"] = loss_value

                if cfg.logging.get("log_wandb"):
                    wandb.log(log_data, step=global_step)

        epoch_time = time() - epoch_start

        log_epoch(
            epoch,
            {
                "loss": total_loss.item(),
                "reg": regl.item(),
                "pred": pl.item(),
                "probe": xy_loss.item(),
            },
            total_epochs=cfg.optim.epochs,
            elapsed_time=epoch_time,
        )

        if cfg.logging.get("log_wandb"):
            wandb.log(
                {"epoch": epoch, "epoch_time": epoch_time},
                step=epoch * len(loader),
            )

        # Save checkpoint
        save_checkpoint(
            latest_ckpt_path,
            model=jepa,
            optimizer=jepa_optimizer,
            scheduler=jepa_scheduler,
            epoch=epoch,
            step=global_step,
            xy_head_state_dict=xy_head.state_dict(),
            probe_optimizer_state_dict=probe_optimizer.state_dict(),
            probe_scheduler_state_dict=probe_scheduler.state_dict(),
        )
        if epoch % cfg.logging.save_every_n_epochs == 0:
            save_checkpoint(
                folder / f"e-{epoch}.pth.tar",
                model=jepa,
                optimizer=jepa_optimizer,
                scheduler=jepa_scheduler,
                epoch=epoch,
                step=global_step,
                xy_head_state_dict=xy_head.state_dict(),
                probe_optimizer_state_dict=probe_optimizer.state_dict(),
                probe_scheduler_state_dict=probe_scheduler.state_dict(),
            )

    logger.info("Training complete.")


if __name__ == "__main__":
    fire.Fire(run)

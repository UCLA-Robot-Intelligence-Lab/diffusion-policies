# If running this script directly, set up environment paths first
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import pathlib
import copy
import random
import shutil
import numpy as np
import torch
import tqdm
import wandb
import hydra

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# -- Imports for the diffusion policy and utilities --
from diffusion_policy.unet_image_policy.unet_image_diffusion_policy import (
    DiffusionUnetImagePolicy,
)
from shared.utils.checkpoint_util import TopKCheckpointManager
from shared.utils.json_logger import JsonLogger
from shared.utils.pytorch_util import dict_apply, optimizer_to
from shared.models.unet.ema_model import EMAModel
from shared.models.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDiffusionUnetImagePolicy:
    def __init__(self, cfg: OmegaConf, output_dir: str = None):
        """
        Initialize the training class with:
         - Hydra config
         - Optional output directory
         - Model, optimizer, EMA (if configured)
         - Random seeds and trackers for training state
        """
        self.cfg = copy.deepcopy(cfg)
        self.output_dir = output_dir or os.getcwd()

        seed = self.cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(self.cfg.policy)

        self.ema_model = None
        if self.cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(
            self.cfg.optimizer, params=self.model.parameters()
        )

        self.global_step = 0
        self.epoch = 0

    def save_checkpoint(self, path=None, tag="latest"):
        """
        Saves a checkpoint with the current training state, including:
         - Global config
         - global_step, epoch
         - Model state dict, optimizer state dict
         - EMA model state if applicable
        """
        if path is None:
            ckpt_dir = pathlib.Path(self.output_dir).joinpath("checkpoints")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = ckpt_dir.joinpath(f"{tag}.ckpt")

        checkpoint = {
            "cfg": self.cfg,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if self.ema_model is not None:
            checkpoint["ema_model_state_dict"] = self.ema_model.state_dict()

        torch.save(checkpoint, path.open("wb"))
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path=None, tag="latest"):
        """
        Loads a checkpoint, restoring:
         - global_step, epoch
         - Model, optimizer, and EMA state
        """
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        if not path.is_file():
            print(f"No checkpoint found at {path} -- skipping load.")
            return

        checkpoint = torch.load(path.open("rb"))
        self.cfg = checkpoint["cfg"]
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.ema_model is not None and "ema_model_state_dict" in checkpoint:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        print(f"Checkpoint loaded from {path}")

    def save_snapshot(self, tag="latest"):
        """
        Saves the entire class instance for quick loads (not long-term).
        Requires the code to remain consistent.
        """
        snapshot_path = pathlib.Path(self.output_dir).joinpath(
            "snapshots", f"{tag}.pkl"
        )
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, snapshot_path.open("wb"))
        return str(snapshot_path.absolute())

    def run(self):
        """
        Main training loop:
         1) Optionally resume from checkpoint
         2) Set up datasets and dataloaders
         3) Configure LR scheduler and EMA
         4) Initialize W&B logging
         5) Perform training, validation, rollout, sampling
        """
        if self.cfg.training.resume:
            last_ckpt = pathlib.Path(self.output_dir).joinpath(
                "checkpoints", "latest.ckpt"
            )
            if last_ckpt.is_file():
                print(f"Resuming from checkpoint {last_ckpt}")
                self.load_checkpoint(path=last_ckpt)

        # WARNING: I am not using the BaseImageDataset!
        train_dataset = hydra.utils.instantiate(self.cfg.task.dataset)
        train_dataloader = DataLoader(train_dataset, **self.cfg.dataloader)

        val_dataset = train_dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **self.cfg.val_dataloader)

        normalizer = train_dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        total_train_steps = (
            len(train_dataloader) * self.cfg.training.num_epochs
        ) // self.cfg.training.gradient_accumulate_every

        lr_scheduler = get_scheduler(
            self.cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.training.lr_warmup_steps,
            num_training_steps=total_train_steps,
            last_epoch=self.global_step - 1,
        )

        ema_obj = None
        if self.cfg.training.use_ema:
            ema_obj = hydra.utils.instantiate(self.cfg.ema, model=self.ema_model)

        # WARNING: I am not using the BaseImageRunner!
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner, output_dir=self.output_dir
        )

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(self.cfg, resolve=True),
            **self.cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **self.cfg.checkpoint.topk,
        )

        device = torch.device(self.cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # Possibly debug mode overrides?
        if self.cfg.training.debug:
            self.cfg.training.num_epochs = 2
            self.cfg.training.max_train_steps = 3
            self.cfg.training.max_val_steps = 3
            self.cfg.training.rollout_every = 1
            self.cfg.training.checkpoint_every = 1
            self.cfg.training.val_every = 1
            self.cfg.training.sample_every = 1

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        train_sampling_batch = None

        # -- TRAINING LOOP --
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(self.cfg.training.num_epochs):
                step_log = dict()

                # ========= Train for this epoch ==========
                if self.cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                # -- TRAINING --
                train_losses = []
                self.model.train()
                loader_desc = f"Training epoch {self.epoch}"
                with tqdm.tqdm(
                    train_dataloader, desc=loader_desc, leave=False
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # Device transfer
                        batch = dict_apply(
                            batch, lambda x: x.to(device, non_blocking=True)
                        )
                        if train_sampling_batch is None:
                            train_sampling_batch = copy.deepcopy(batch)

                        # Compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / self.cfg.training.gradient_accumulate_every
                        loss.backward()

                        # Step optimizer
                        if (
                            self.global_step
                            % self.cfg.training.gradient_accumulate_every
                            == 0
                        ):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # Update EMA
                        if ema_obj is not None:
                            ema_obj.step(self.model)

                        # Logging
                        raw_loss_val = raw_loss.item()
                        train_losses.append(raw_loss_val)
                        tepoch.set_postfix(loss=raw_loss_val, refresh=False)
                        step_log = {
                            "train_loss": raw_loss_val,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            # Log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        # Early stop for debug
                        if (
                            self.cfg.training.max_train_steps is not None
                            and batch_idx >= (self.cfg.training.max_train_steps - 1)
                        ):
                            break

                # At the end of each epoch, replace train_loss with the epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= Eval for this epoch ==========
                if self.ema_model is not None:
                    policy = self.ema_model
                else:
                    policy = self.model
                policy.eval()

                # Run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    step_log.update(runner_log)

                # -- VALIDATION --
                if (self.epoch % self.cfg.training.val_every) == 0:
                    val_losses = []
                    with torch.no_grad():
                        loader_desc = f"Validation epoch {self.epoch}"
                        with tqdm.tqdm(
                            val_dataloader, desc=loader_desc, leave=False
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(
                                    batch, lambda x: x.to(device, non_blocking=True)
                                )
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss.item())

                                if (
                                    self.cfg.training.max_val_steps is not None
                                    and batch_idx
                                    >= (self.cfg.training.max_val_steps - 1)
                                ):
                                    break

                    if len(val_losses) > 0:
                        val_loss = float(np.mean(val_losses))
                        step_log["val_loss"] = val_loss

                # -- SAMPLING --
                if (
                    self.epoch % self.cfg.training.sample_every
                ) == 0 and train_sampling_batch:
                    with torch.no_grad():
                        batch = dict_apply(
                            train_sampling_batch,
                            lambda x: x.to(device, non_blocking=True),
                        )
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]

                        result = policy.predict_action(obs_dict)
                        pred_action = result["action_pred"]

                        mse_error = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse_error.item()

                # -- CHECKPOINTING --
                if (self.epoch % self.cfg.training.checkpoint_every) == 0:
                    if self.cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()

                    if self.cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
                    topk_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_path is not None:
                        self.save_checkpoint(path=topk_path)

                # ========= Eval end for this epoch ==========
                policy.train()

                # End of epoch, log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)

                self.global_step += 1
                self.epoch += 1
                # May the gradients flow :)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    # Hydra entry point. Instantiates and runs the training process.
    trainer = TrainDiffusionUnetImagePolicy(cfg)
    trainer.run()


if __name__ == "__main__":
    main()

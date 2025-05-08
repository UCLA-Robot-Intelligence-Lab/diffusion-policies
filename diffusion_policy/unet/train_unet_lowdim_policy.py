if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)


import os
import pathlib
import hydra
import copy
import dill
import torch
import threading
import sys
import random
import wandb
import tqdm
import numpy as np
import time
import matplotlib

matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from hydra.core.hydra_config import HydraConfig
from hydra import initialize, compose
from omegaconf import OmegaConf
from typing import Optional
from torch.utils.data import DataLoader

from diffusion_policy.unet.unet_lowdim_policy import DiffusionUnetLowdimPolicy
from shared.utils.checkpoint_util import TopKCheckpointManager
from shared.utils.json_logger import JsonLogger
from shared.utils.pytorch_util import (
    dict_apply,
    optimizer_to,
    copy_to_cpu,
    temporary_attribute,
)
from shared.models.unet.ema_model import EMAModel
from shared.models.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDiffusionUnetLowdimWorkspace:
    include_keys = ("global_step", "epoch")
    exclude_keys = ()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None

        # Set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Debug print
        print(f"Creating model with: obs_dim={cfg.obs_dim}, action_dim={cfg.action_dim}, num_obs_steps={cfg.num_obs_steps}")
        print(f"Expected global cond dim: {cfg.obs_dim * cfg.num_obs_steps}, cond_dim_G in config: {cfg.policy.model.cond_dim_G}")
        
        # Configure model
        self.model: DiffusionUnetLowdimPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: Optional[DiffusionUnetLowdimPolicy] = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # Configure optimizer
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )

        # Initialize training state
        self.global_step = 0
        self.epoch = 0

        # Inference step testing
        self.num_inference_steps_ls = [10, 16, 25, 50, 100]
        self.coverage_data = {
            "train": {s: [] for s in self.num_inference_steps_ls},
            "test": {s: [] for s in self.num_inference_steps_ls},
        }

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def update_coverage_plot(self):
        coverage_fig, coverage_ax = plt.subplots(figsize=(10, 5))
        coverage_ax.set_title("Coverage vs Num Inference Steps")
        coverage_ax.set_xlabel("Inference Steps")
        coverage_ax.set_ylabel("Coverage")

        coverage_avgs_train = []
        coverage_avgs_test = []
        coverage_max_train = []
        coverage_max_test = []

        for s in self.num_inference_steps_ls:
            train_values = self.coverage_data["train"][s]
            test_values = self.coverage_data["test"][s]

            if len(train_values) > 0:
                coverage_avgs_train.append(np.mean(train_values))
                coverage_max_train.append(np.max(train_values))
            else:
                coverage_avgs_train.append(0.0)
                coverage_max_train.append(0.0)

            if len(test_values) > 0:
                coverage_avgs_test.append(np.mean(test_values))
                coverage_max_test.append(np.max(test_values))
            else:
                coverage_avgs_test.append(0.0)
                coverage_max_test.append(0.0)

        x_positions = np.arange(len(self.num_inference_steps_ls))
        bar_width = 0.2

        # 1) Train Avg
        coverage_ax.bar(
            x_positions - 1.5 * bar_width,
            coverage_avgs_train,
            bar_width,
            label="Train Avg",
            color="C0",
            alpha=0.7,
        )
        # 2) Test Avg
        coverage_ax.bar(
            x_positions - 0.5 * bar_width,
            coverage_avgs_test,
            bar_width,
            label="Test Avg",
            color="C1",
            alpha=0.7,
        )
        # 3) Train Max
        coverage_ax.bar(
            x_positions + 0.5 * bar_width,
            coverage_max_train,
            bar_width,
            label="Train Max",
            color="C2",
            alpha=0.7,
        )
        # 4) Test Max
        coverage_ax.bar(
            x_positions + 1.5 * bar_width,
            coverage_max_test,
            bar_width,
            label="Test Max",
            color="C3",
            alpha=0.7,
        )

        coverage_ax.set_xticks(x_positions)
        coverage_ax.set_xticklabels([str(s) for s in self.num_inference_steps_ls])

        all_values = (
            coverage_avgs_train
            + coverage_avgs_test
            + coverage_max_train
            + coverage_max_test
        )
        if len(all_values) > 0:
            max_cov = max(all_values)
            coverage_ax.set_ylim([0, max(max_cov, 1e-6) * 1.1])
        coverage_ax.legend()

        coverage_fig.tight_layout()
        wandb.log({"coverage_plot": wandb.Image(coverage_fig)}, step=self.global_step)
        plt.close(coverage_fig)

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # Resume training if specified
        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)

        # Configure dataset
        dataset = hydra.utils.instantiate(cfg.tasks.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # Configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        """
        Learning rate scheduler updates learning rate dynamically during
        training.
        """
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        """
        EMA (exponential moving average) maintains a smoothed
        version of the model weights. Shown to improve performance,
        and the weighting of the weights favors the last few epochs.
        """
        ema: Optional[EMAModel] = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        """
        The environment runner handles simulating the environment.
        """
        env_runner = hydra.utils.instantiate(
            cfg.tasks.env_runner, output_dir=self.output_dir
        )

        # Configure logging with Weights & Biases
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})

        """
        TopK Checkpoint manager works by maintaining top-K model
        checkpoints based on performance.
        """
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk
        )

        # Device setup
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # Save batch for sampling
        train_sampling_batch = None

        # Debug mode adjustments
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # Training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = {}
                # ========= Train for this epoch ==========
                train_losses = []
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # Device transfer
                        batch = dict_apply(
                            batch, lambda x: x.to(device, non_blocking=True)
                        )
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # Compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # Step optimizer
                        if (
                            self.global_step % cfg.training.gradient_accumulate_every
                            == 0
                        ):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # Update EMA
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # Logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            # Log intermediate steps
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if cfg.training.max_train_steps is not None and batch_idx >= (
                            cfg.training.max_train_steps - 1
                        ):
                            break

                # At the end of each epoch, replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= Evaluation for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # Run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # Log all
                    step_log.update(runner_log)

                # Coverage logging, track separately for train vs test
                if cfg.training.get("measure_coverage", False) and (
                    self.epoch % cfg.training.rollout_every == 0
                ):
                    coverage_log_dict = {}
                    for s in self.num_inference_steps_ls:
                        with temporary_attribute(policy, "num_inference_steps", s):
                            runner_log_s = env_runner.run(policy)
                        for k, v in runner_log_s.items():
                            if "sim_max_coverage_" in k:
                                coverage_val = float(v)
                                if k.startswith("train"):
                                    self.coverage_data["train"][s].append(coverage_val)
                                    coverage_log_dict[f"train_{s}_coverage/{k}"] = (
                                        coverage_val
                                    )
                                elif k.startswith("test"):
                                    self.coverage_data["test"][s].append(coverage_val)
                                    coverage_log_dict[f"train_{s}_coverage/{k}"] = (
                                        coverage_val
                                    )

                    self.update_coverage_plot()
                    wandb.log(coverage_log_dict, step=self.global_step)

                # Run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = []
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(
                                    batch, lambda x: x.to(device, non_blocking=True)
                                )
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (
                                    cfg.training.max_val_steps is not None
                                    and batch_idx >= (cfg.training.max_val_steps - 1)
                                ):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # Log epoch average validation loss
                            step_log["val_loss"] = val_loss

                # Run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # Sample trajectory from training set, and evaluate difference
                        batch = dict_apply(
                            train_sampling_batch,
                            lambda x: x.to(device, non_blocking=True),
                        )
                        obs_dict = {'obs': batch['obs']}
                        gt_action = batch["action"]

                        t0 = time.time()
                        result = policy.predict_action(obs_dict)
                        t1 = time.time()

                        pred_action = result["action_pred"]

                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()

                        # Track time elapsed for diffusion policy
                        elapsed_time = t1 - t0
                        step_log["time_elapsed_predict_action"] = elapsed_time

                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # Checkpointing
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # Save checkpoints and snapshots
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # Sanitize metric names
                    metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}

                    # Manage Top-K checkpoints
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                # ========= End of Evaluation ==========
                policy.train()

                # End of epoch logging
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

    def save_checkpoint(
        self,
        path=None,
        tag="latest",
        exclude_keys=None,
        include_keys=None,
        use_thread=True,
    ):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cfg": self.cfg, "state_dicts": {}, "pickles": {}}

        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                # Modules, optimizers, etc.
                if key not in exclude_keys:
                    if use_thread:
                        payload["state_dicts"][key] = copy_to_cpu(value.state_dict())
                    else:
                        payload["state_dicts"][key] = value.state_dict()
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open("wb"), pickle_module=dill)
            )
            self._saving_thread.start()
        else:
            torch.save(payload, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    def get_checkpoint_path(self, tag="latest"):
        return pathlib.Path(self.output_dir).joinpath("checkpoints", f"{tag}.ckpt")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = ()
        if include_keys is None:
            include_keys = payload["pickles"].keys()

        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])

    def load_checkpoint(
        self, path=None, tag="latest", exclude_keys=None, include_keys=None, **kwargs
    ):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open("rb"), pickle_module=dill, **kwargs)
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload

    def save_snapshot(self, tag="latest"):
        """
        Quick loading and saving for research, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath("snapshots", f"{tag}.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path.open("wb"), pickle_module=dill)
        return str(path.absolute())

    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, "rb"), pickle_module=dill)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name="train_unet_lowdim_real_policy",
)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main() 
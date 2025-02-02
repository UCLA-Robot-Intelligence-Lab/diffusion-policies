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
from typing import Optional, Dict, List
from torch.utils.data import DataLoader
from collections import defaultdict

from shortcut_policy.unet_shortcut.unet_image_shortcut_policy import (
    UnetImageShortcutPolicy,
)

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


class TrainDiffusionUnetImageWorkspace:
    include_keys = ("global_step", "epoch")
    exclude_keys = ()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        self.cfg = cfg
        self._output_dir = cfg.output_dir if "output_dir" in cfg else output_dir
        self._saving_thread = None

        # Set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Create the model
        self.model: UnetImageShortcutPolicy = hydra.utils.instantiate(cfg.policy)

        # Optionally create an EMA copy
        self.ema_model: Optional[UnetImageShortcutPolicy] = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # Configure optimizer
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()
        )

        # Initialize training state
        self.global_step = 0
        self.epoch = 0

        # Shortcut steps we want to test
        self.num_shortcut_steps = [1, 2, 4, 8, 16, 32, 64, 128]

        # For logging speed vs. MSE
        self.training_steps: List[int] = []
        self.time_data = {f"Time_{s}": [] for s in self.num_shortcut_steps}
        self.mse_data = {f"MSE_{s}": [] for s in self.num_shortcut_steps}
        self.plot_log_interval = 1

        # For coverage logging (track train and test separately)
        # We'll store all coverage values to later compute both average & max
        self.coverage_data = {
            "train": {s: [] for s in self.num_shortcut_steps},
            "test": {s: [] for s in self.num_shortcut_steps},
        }

        # Output dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory set to: {self.output_dir}")

        # Speed & MSE figure handles
        self.speed_fig, self.speed_ax = plt.subplots(figsize=(10, 6))
        self.speed_ax.set_xlabel("Global Step")
        self.speed_ax.set_ylabel("Time Elapsed (s)")
        self.speed_ax.set_title("Shortcut Policy speed test across step sizes")
        self.speed_path = os.path.join(self.output_dir, "shortcut_speed_test.png")

        self.mse_fig, self.mse_ax = plt.subplots(figsize=(10, 6))
        self.mse_ax.set_xlabel("Global Step")
        self.mse_ax.set_ylabel("Mean Squared Error")
        self.mse_ax.set_title("MSE across step sizes")
        self.mse_path = os.path.join(self.output_dir, "shortcut_mse_per_num_steps.png")

    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir

    def update_plot(self):
        """Update & log the speed and MSE plots for the current self.global_step."""
        custom_plots = []
        # ---- SPEED PLOT ----
        self.speed_ax.clear()
        self.speed_ax.set_xlabel("Global Step")
        self.speed_ax.set_ylabel("Time Elapsed (s)")
        self.speed_ax.set_title("Shortcut Policy speed test across step sizes")
        speed_legend_handles = []
        for s in self.num_shortcut_steps:
            steps = self.training_steps
            times = self.time_data[f"Time_{s}"]
            if len(steps) > 1:
                self.speed_ax.plot(
                    steps,
                    times,
                    linestyle="-",
                    alpha=0.5,
                    color=f"C{self.num_shortcut_steps.index(s)}",
                )
            self.speed_ax.scatter(
                steps, times, s=20, color=f"C{self.num_shortcut_steps.index(s)}"
            )
            speed_legend_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=f"C{self.num_shortcut_steps.index(s)}",
                    marker="o",
                    linestyle="",
                    label=f"Steps={s}",
                )
            )
        self.speed_ax.legend(
            handles=speed_legend_handles, fontsize=8, loc="upper right"
        )

        all_times = []
        for sublist in self.time_data.values():
            all_times.extend(sublist)
        if len(all_times) > 1:
            min_t, max_t = min(all_times), max(all_times)
            if min_t < max_t:
                self.speed_ax.set_yticks(np.linspace(min_t, max_t, num=5))

        self.speed_fig.tight_layout()
        custom_plots.append(wandb.Image(self.speed_fig))

        # ---- MSE PLOT ----
        self.mse_ax.clear()
        self.mse_ax.set_xlabel("Global Step")
        self.mse_ax.set_ylabel("Mean Squared Error")
        self.mse_ax.set_title("MSE across step sizes")
        mse_legend_handles = []
        for s in self.num_shortcut_steps:
            steps = self.training_steps
            mse_vals = self.mse_data[f"MSE_{s}"]
            if len(steps) > 1:
                self.mse_ax.plot(
                    steps,
                    mse_vals,
                    linestyle="-",
                    alpha=0.5,
                    color=f"C{self.num_shortcut_steps.index(s)}",
                )
            self.mse_ax.scatter(
                steps, mse_vals, s=20, color=f"C{self.num_shortcut_steps.index(s)}"
            )
            mse_legend_handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=f"C{self.num_shortcut_steps.index(s)}",
                    marker="o",
                    linestyle="",
                    label=f"Steps={s}",
                )
            )
        self.mse_ax.legend(handles=mse_legend_handles, fontsize=8, loc="upper right")

        all_mse_vals = []
        for sublist in self.mse_data.values():
            all_mse_vals.extend(sublist)
        if len(all_mse_vals) > 1:
            min_m, max_m = min(all_mse_vals), max(all_mse_vals)
            if min_m < max_m:
                self.mse_ax.set_yticks(np.linspace(min_m, max_m, num=5))

        self.mse_fig.tight_layout()
        custom_plots.append(wandb.Image(self.mse_fig))

        # Log both in one go
        wandb.log({"custom_plots": custom_plots}, step=self.global_step)

    def update_coverage_plot(self):
        coverage_fig, coverage_ax = plt.subplots(figsize=(10, 5))
        coverage_ax.set_title("Coverage vs Num Inference Steps")
        coverage_ax.set_xlabel("Inference Steps")
        coverage_ax.set_ylabel("Coverage")

        coverage_avgs_train = []
        coverage_avgs_test = []
        coverage_max_train = []
        coverage_max_test = []

        for s in self.num_shortcut_steps:
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

        x_positions = np.arange(len(self.num_shortcut_steps))
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
        coverage_ax.set_xticklabels([str(s) for s in self.num_shortcut_steps])

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

        # Set normalizer
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        # LR Scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        # EMA
        ema: Optional[EMAModel] = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # Env runner
        env_runner = hydra.utils.instantiate(
            cfg.tasks.env_runner, output_dir=self.output_dir
        )

        # Init W&B
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})

        # TopK manager
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk
        )

        # Device
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None

        # Debug adjustments
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):

                step_log = {}
                # ---------------------------
                # 1) TRAIN
                # ---------------------------
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = []
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(
                            batch, lambda x: x.to(device, non_blocking=True)
                        )
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if (
                            self.global_step % cfg.training.gradient_accumulate_every
                        ) == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if cfg.training.use_ema and ema is not None:
                            ema.step(self.model)

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
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and (
                            batch_idx >= (cfg.training.max_train_steps - 1)
                        ):
                            break

                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ---------------------------
                # 2) EVAL (Rollout & Coverage)
                # ---------------------------
                policy = self.model if not cfg.training.use_ema else self.ema_model
                policy.eval()

                # Possibly rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    step_log.update(runner_log)

                # Coverage logging, track separately for train vs test
                if (
                    cfg.training.get("measure_coverage", False)
                    and (self.epoch % cfg.training.rollout_every) == 0
                ):
                    coverage_log_dict = {}
                    for s in self.num_shortcut_steps:
                        with temporary_attribute(policy, "num_inference_steps", s):
                            # env_runner might produce multiple coverage keys,
                            # e.g. "train/sim_max_coverage_X" or "test/sim_max_coverage_X"
                            runner_log_s = env_runner.run(policy)
                        for k, v in runner_log_s.items():
                            if "sim_max_coverage_" in k:
                                coverage_val = float(v)
                                # Decide which side (train/test) to store
                                if k.startswith("train"):
                                    self.coverage_data["train"][s].append(coverage_val)
                                    coverage_log_dict[f"train_{s}_coverage/{k}"] = (
                                        coverage_val
                                    )
                                elif k.startswith("test"):
                                    self.coverage_data["test"][s].append(coverage_val)
                                    coverage_log_dict[f"test_{s}_coverage/{k}"] = (
                                        coverage_val
                                    )

                    # Update the grouped bar chart for coverage
                    self.update_coverage_plot()
                    # Also log raw coverage values
                    wandb.log(coverage_log_dict, step=self.global_step)

                # Validation
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
                            step_log["val_loss"] = val_loss

                # ---------------------------
                # 3) SAMPLING & SHORTCUT TEST
                # ---------------------------
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = dict_apply(
                            train_sampling_batch,
                            lambda x: x.to(device, non_blocking=True),
                        )
                        obs_dict = batch["obs"]
                        gt_action = batch["action"]

                        # Normal sampling
                        result = policy.predict_action_shortcut(obs_dict)
                        pred_action = result["action_pred"]
                        mse_val = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse_val.item()

                        # Speed & MSE
                        for s in self.num_shortcut_steps:
                            t0 = time.time()
                            result_n = policy.predict_action_shortcut(obs_dict, s)
                            t1 = time.time()

                            pred_n = result_n["action_pred"]
                            mse_n = torch.nn.functional.mse_loss(pred_n, gt_action)

                            self.time_data[f"Time_{s}"].append(t1 - t0)
                            self.mse_data[f"MSE_{s}"].append(mse_n.item())

                        self.training_steps.append(self.global_step)
                        if (self.global_step % self.plot_log_interval) == 0:
                            self.update_plot()

                        del batch, obs_dict, gt_action, result, pred_action, mse_val

                # ---------------------------
                # 4) CHECKPOINT
                # ---------------------------
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                # Return to train mode
                policy.train()

                # Log final step metrics
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
            # Save state_dicts for modules like model, optimizer, etc.
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                if key not in exclude_keys:
                    if use_thread:
                        payload["state_dicts"][key] = copy_to_cpu(value.state_dict())
                    else:
                        payload["state_dicts"][key] = value.state_dict()
            # Save certain attributes as pickles
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
        self,
        path=None,
        tag="latest",
        exclude_keys=None,
        include_keys=None,
        **kwargs,
    ):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open("rb"), pickle_module=dill, **kwargs)
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=include_keys)
        return payload

    def save_snapshot(self, tag="latest"):
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
    config_name="train_unet_image_shortcut_policy",
)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()

import torch
import robomimic.utils.obs_utils as ObsUtils

from typing import Dict
from shared.models.common.normalizer import LinearNormalizer
from shared.utils.pytorch_util import dict_apply
from shared.utils.robomimic_config_util import get_robomimic_config

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo


class RobomimicImagePolicy(torch.nn.Module):
    def __init__(
        self,
        shape_meta: dict,
        algo_name="bc_rnn",
        obs_type="image",
        task_name="square",
        dataset_type="ph",
        crop_shape=(76, 76),
    ):
        super(RobomimicImagePolicy, self).__init__()

        # parse shape_meta
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]
        obs_config = {"low_dim": [], "rgb": [], "depth": [], "scan": []}
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr["shape"]
            obs_key_shapes[key] = list(shape)

            obs_type_key = attr.get("type", "low_dim")
            if obs_type_key == "rgb":
                obs_config["rgb"].append(key)
            elif obs_type_key == "low_dim":
                obs_config["low_dim"].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type_key}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name=algo_name,
            hdf5_type=obs_type,
            task_name=task_name,
            dataset_type=dataset_type,
        )

        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality["obs_randomizer_class"] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == "CropRandomizer":
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        model: PolicyAlgo = algo_factory(
            algo_name=config.algo_name,
            config=config,
            obs_key_shapes=obs_key_shapes,
            ac_dim=action_dim,
            device="cpu",
        )

        self.model = model
        self.nets = model.nets
        self.normalizer = LinearNormalizer()
        self.config = config

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        if device is not None:
            self.model.device = device
        return super().to(*args, **kwargs)

    # =========== inference =============
    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        nobs_dict = self.normalizer(obs_dict)
        robomimic_obs_dict = dict_apply(nobs_dict, lambda x: x[:, 0, ...])
        naction = self.model.get_action(robomimic_obs_dict)
        action = self.normalizer["action"].unnormalize(naction)
        # (B, Da) → (B, 1, Da)
        result = {"action": action[:, None, :]}
        return result

    def reset(self):
        self.model.reset()

    # =========== training ==============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def train_on_batch(self, batch, epoch, validate=False):
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        robomimic_batch = {"obs": nobs, "actions": nactions}
        input_batch = self.model.process_batch_for_training(robomimic_batch)
        info = self.model.train_on_batch(
            batch=input_batch, epoch=epoch, validate=validate
        )
        return info

    def on_epoch_end(self, epoch):
        self.model.on_epoch_end(epoch)

    def get_optimizer(self):
        return self.model.optimizers["policy"]

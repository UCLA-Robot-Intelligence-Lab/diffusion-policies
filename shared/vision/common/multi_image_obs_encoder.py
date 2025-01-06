import copy
import torch
import torch.nn as nn
import torchvision

from typing import Dict, Tuple, Union
from shared.vision.common.crop_randomizer import CropRandomizer
from shared.utils.pytorch_util import replace_submodules


class ObsEncoder(nn.Module):
    def __init__(
        self,
        shape_meta: dict,
        vision_backbone: Union[nn.Module, Dict[str, nn.Module]],
        resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        random_crop: bool = True,
        use_group_norm: bool = False,
        share_vision_backbone: bool = False,
        imagenet_norm: bool = False,
    ):
        """
        args:
            shape_meta : Dictionary containing metadata for the input/output
                         shapes for our model, shapes specific to each task.
                         Contains 'obs' key with nested dictionary containing
                         'image' and 'agent_pos' keys with relevant shapes.
                         Contains 'action' key with action dimension shape.
            vision_backbone : What visual encoder to use, either as a torch
                              network module or a dictionary with loaded weights.
            resize_shape : For images, what to resize (down) the image to.
            crop_shape : Shape of the images after cropping.
            random-cop : Flag to determine to use the crop_randomizer instead of
                         default center crop.
            use_group_norm : Flag to determine whether to replace 2D Batch Norm
                             for resnet backbones with Group Norm.
            share_vision_backbone : Flag to determine whether to use the same
                                    visual encoder for each camera (stream of
                                    observations) or multple.
            imagenet_norm : Flag to determine whether to renormalize image with
                            imagenet normalization. Assumes input is in [0, 1]
        """
        super().__init__()

        rgb_keys = []
        low_dim_keys = []
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        if share_vision_backbone:
            assert isinstance(vision_backbone, nn.Module)
            key_model_map["rgb"] = vision_backbone

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            if type == "rgb":
                rgb_keys.append(key)
                this_model = None
                if not share_vision_backbone:
                    if isinstance(vision_backbone, dict):
                        this_model = vision_backbone[key]
                    else:
                        assert isinstance(vision_backbone, nn.Module)
                        this_model = copy.deepcopy(vision_backbone)

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features // 16,
                                num_channels=x.num_features,
                            ),
                        )
                    key_model_map[key] = this_model

                # Handle resizing.
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                # Setup crop randomizer.
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False,
                        )
                    else:
                        this_normalizer = torchvision.transforms.CenterCrop(size=(h, w))
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )

                this_transform = nn.Sequential(
                    this_resizer, this_randomizer, this_normalizer
                )
                key_transform_map[key] = this_transform
            elif type == "low_dim":
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_vision_backbone = share_vision_backbone
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.dummy_parameter = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def forward(self, obs_dict):
        share_vision_backbone = self.share_vision_backbone
        rgb_keys = self.rgb_keys
        low_dim_keys = self.low_dim_keys
        key_shape_map = self.key_shape_map
        key_model_map = self.key_model_map
        key_transform_map = self.key_transform_map
        batch_size = None
        features = []

        if share_vision_backbone:
            imgs = []
            for key in rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == key_shape_map[key]
                img = key_transform_map[key](img)
                imgs.append(img)

            # [ N*B, C, H, W ] -> [ N*B, O ] -> [ B, O*N ]
            imgs = torch.cat(imgs, dim=0)
            feature = key_model_map["rgb"](imgs)
            feature = feature.reshape(-1, batch_size, *feature.shape[1:])
            feature = torch.moveaxis(feature, 0, 1)
            feature = feature.reshape(batch_size, -1)

            features.append(feature)
        else:
            for key in rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == key_shape_map[key]
                img = key_transform_map[key](img)
                feature = key_model_map[key](img)
                features.append(feature)

        for key in low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == key_shape_map[key]
            features.append(data)

        result = torch.cat(features, dim=-1)
        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta["obs"]
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            this_obs = torch.zeros(
                (batch_size,) + shape, dtype=self.dtype, device=self.device
            )
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape

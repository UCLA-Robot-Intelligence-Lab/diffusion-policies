import copy
import torch
import torch.nn as nn
import torchvision.transforms as T
from shared.encoders.resnet.resnet_18 import get_resnet18
from shared.encoders.common.crop_randomizer import CropRandomizer
from shared.utils.pytorch_util import replace_submodules
from typing import Dict, Tuple, Union


class ObsEncoder(nn.Module):
    """
    Encodes multi-modal observations, including RGB images and low-dimensional inputs,
    into a unified feature representation.
    """

    def __init__(
        self,
        shape_meta: Dict[
            str, Dict[str, Union[Dict[str, Tuple[int, ...]], Tuple[int, ...]]]
        ],
        vision_backbone: Union[nn.Module, Dict[str, nn.Module]] = None,
        resize_shape: Union[Tuple[int, int], Dict[str, Tuple[int, int]], None] = None,
        crop_shape: Union[Tuple[int, int], Dict[str, Tuple[int, int]], None] = None,
        random_crop: bool = True,
        imagenet_norm: bool = True,
        use_group_norm: bool = False,
        share_vision_backbone: bool = True,
    ):
        super().__init__()
        # print("ObsEncoder initialized with shape_meta:", shape_meta)
        self.shape_meta = shape_meta
        self.share_vision_backbone = share_vision_backbone
        self.rgb_keys = []
        self.low_dim_keys = []
        self.key_model_map = nn.ModuleDict()
        self.key_transform_map = nn.ModuleDict()
        self.key_shape_map = {}

        # Handle shared vision backbone
        if share_vision_backbone:
            if vision_backbone is None:
                vision_backbone = get_resnet18(pretrained=True)  # Default to ResNet18
            assert isinstance(vision_backbone, nn.Module)
            self.key_model_map["shared_rgb"] = vision_backbone

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            obs_type = attr.get("type", "low_dim")
            self.key_shape_map[key] = shape

            if obs_type == "rgb":
                self.rgb_keys.append(key)
                # Configure model for this key
                if not share_vision_backbone:
                    if isinstance(vision_backbone, dict):
                        self.key_model_map[key] = vision_backbone[key]
                    else:
                        assert isinstance(vision_backbone, nn.Module)
                        self.key_model_map[key] = copy.deepcopy(vision_backbone)

                # Apply GroupNorm if specified
                # removed since it was causing issues and by defualt our resnet comes with GN

                # Configure resizing and cropping
                resizer = nn.Identity()
                if resize_shape:
                    h, w = (
                        resize_shape[key]
                        if isinstance(resize_shape, dict)
                        else resize_shape
                    )
                    resizer = T.Resize(size=(h, w))
                    shape = (shape[0], h, w)

                randomizer = nn.Identity()
                if crop_shape:
                    h, w = (
                        crop_shape[key] if isinstance(crop_shape, dict) else crop_shape
                    )
                    if random_crop:
                        randomizer = CropRandomizer(
                            input_shape=shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False,
                        )
                    else:
                        randomizer = T.CenterCrop(size=(h, w))

                normalizer = nn.Identity()
                if imagenet_norm:
                    normalizer = T.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )

                self.key_transform_map[key] = nn.Sequential(
                    resizer, randomizer, normalizer
                )

            elif obs_type == "low_dim":
                self.low_dim_keys.append(key)
            else:
                raise ValueError(f"Unsupported observation type: {obs_type}")

        # Sort keys for consistent ordering
        self.rgb_keys = sorted(self.rgb_keys)
        self.low_dim_keys = sorted(self.low_dim_keys)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = None
        features = []

        if self.share_vision_backbone:
            # Pass all RGB observations to the shared vision model
            imgs = []
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                img = self.key_transform_map[key](img)
                imgs.append(img)

            # Process concatenated images
            imgs = torch.cat(imgs, dim=0)
            feature = self.key_model_map["shared_rgb"](imgs)
            feature = (
                feature.view(len(self.rgb_keys), batch_size, -1)
                .transpose(0, 1)
                .contiguous()
            )
            features.append(feature.view(batch_size, -1))
        else:
            # Pass each RGB observation to its own model
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)

        # Add low-dimensional features
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            features.append(data)

        # Concatenate features
        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def output_shape(self) -> Tuple[int, ...]:
        example_obs = {
            key: torch.zeros((1,) + shape, dtype=torch.float32)
            for key, shape in self.key_shape_map.items()
        }
        output = self.forward(example_obs)
        return output.shape[1:]


def test_obs_encoder():
    B, C, H, W = 4, 3, 256, 256
    D = 10
    crop_height, crop_width = 112, 112

    # Updated shape_meta structure with nested observations
    shape_meta = {
        "obs": {
            "image": {"shape": (C, H, W), "type": "rgb"},  # RGB observation
            "agent_pos": {
                "shape": (D,),
                "type": "low_dim",
            },  # Low-dimensional observation
        },
        "action": {"shape": (D,), "type": "low_dim"},  # Action metadata
    }

    # Flattened observation dictionary to match expected structure
    obs_dict = {
        "image": torch.randn(B, C, H, W),  # Flattened RGB input
        "agent_pos": torch.randn(B, D),  # Flattened low-dimensional input
    }

    # Initialize the encoder
    encoder = ObsEncoder(
        shape_meta=shape_meta,
        vision_backbone=None,  # Use default ResNet18
        resize_shape=(224, 224),
        random_crop=True,
        crop_shape=(crop_height, crop_width),
        imagenet_norm=True,
    )

    # Perform forward pass
    output = encoder(obs_dict)

    # Assertions for testing
    assert isinstance(output, torch.Tensor), "Output is not a torch.Tensor."
    assert output.shape[0] == B, "Output batch size mismatch."
    print("Output shape:", output.shape)

    output_shape = encoder.output_shape()
    print("Expected output shape:", output_shape)

    print("test_obs_encoder passed!")


if __name__ == "__main__":
    test_obs_encoder()

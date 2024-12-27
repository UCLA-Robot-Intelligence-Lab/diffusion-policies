from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import torchvision.transforms as T
from resnet_18 import get_resnet18

"""
Dimension key:

B: batch size
C: input channels / dimensions (e.g., RGB image channels)
H: image height
W: image width
D: output feature dimensions
L: low-dimensional feature dimensions

"""


class ObsEncoder(nn.Module):
    """
    Encodes multi-modal observations, including RGB images and low-dimensional inputs,
    into a unified feature representation.
    """

    def __init__(
        self,
        shape_meta: Dict[str, Dict[str, Tuple[int, ...]]],
        vision_backbone: nn.Module = None,
        resize_shape: Tuple[int, int] = (224, 224),
        imagenet_norm: bool = True,
        share_vision_backbone: bool = True,
    ):
        """
        Args:
            shape_meta: Metadata describing input observation shapes and types.
            vision_backbone: Pretrained vision model (e.g., ResNet). Defaults to ResNet18.
            resize_shape: Tuple specifying height and width for image resizing.
            imagenet_norm: If True, applies ImageNet normalization to RGB inputs.
            share_vision_backbone: If True, all RGB inputs share the same vision model.
        """
        super().__init__()
        self.shape_meta = shape_meta
        self.share_vision_backbone = share_vision_backbone
        self.resize_shape = resize_shape
        self.imagenet_norm = imagenet_norm

        # Initialize vision backbone
        self.vision_backbone = vision_backbone or get_resnet18(
            pretrained=True, num_classes=512
        )
        self.rgb_keys = [
            key for key, meta in shape_meta.items() if meta["type"] == "rgb"
        ]
        self.low_dim_keys = [
            key for key, meta in shape_meta.items() if meta["type"] == "low_dim"
        ]

        # Apparently this is the stanford torchvision pipeline?
        self.image_transform = T.Compose(
            [
                T.Resize(resize_shape),
                (
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    if imagenet_norm
                    else nn.Identity()
                ),
            ]
        )

        # Create a dictionary of models for each RGB input if models are not shared
        if not share_vision_backbone:
            self.key_model_map = nn.ModuleDict(
                {
                    key: get_resnet18(pretrained=True, num_classes=512)
                    for key in self.rgb_keys
                }
            )
        else:
            self.key_model_map = nn.ModuleDict({"shared_rgb": self.vision_backbone})

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encodes observations into a unified feature tensor.

        Args:
            obs_dict: Dictionary of input observations, with keys matching `shape_meta`.

        Returns:
            torch.Tensor: Unified feature tensor of shape [B, D].
        """
        batch_size = None
        features = []

        for key in self.rgb_keys:
            img_BCHW = obs_dict[key]
            if batch_size is None:
                batch_size = img_BCHW.shape[0]
            else:
                assert (
                    batch_size == img_BCHW.shape[0]
                ), "Inconsistent batch size across inputs."

            img_BCHW = self.image_transform(img_BCHW)
            model = (
                self.key_model_map["shared_rgb"]
                if self.share_vision_backbone
                else self.key_model_map[key]
            )
            features.append(model(img_BCHW))

        # Process low-dimensional inputs
        for key in self.low_dim_keys:
            low_dim_BL = obs_dict[key]
            if batch_size is None:
                batch_size = low_dim_BL.shape[0]
            else:
                assert (
                    batch_size == low_dim_BL.shape[0]
                ), "Inconsistent batch size across inputs."

            features.append(low_dim_BL)

        return torch.cat(features, dim=-1)

    def output_shape(self) -> Tuple[int, ...]:
        """
        Calculates the output shape of the encoder based on metadata.

        Returns:
            Tuple[int, ...]: Shape of the encoded feature vector.
        """
        # Use example inputs to determine output shape
        example_obs = {
            key: torch.zeros(1, *meta["shape"]) for key, meta in self.shape_meta.items()
        }
        with torch.no_grad():
            output = self.forward(example_obs)
        return output.shape[1:]


# ====== TEST FUNCTION ======
# This is just forward pass tests for the vision backbone
# Right now it only supports ResNet18 (as tested)
# At some point, maybe we can experiment with other backbones, i.e., ViT
def main():
    # Test case with both RGB and low-dimensional inputs
    shape_meta = {
        "rgb1": {"shape": (3, 224, 224), "type": "rgb"},
        "rgb2": {"shape": (3, 224, 224), "type": "rgb"},
        "low_dim": {"shape": (10,), "type": "low_dim"},
    }

    encoder = ObsEncoder(shape_meta=shape_meta, share_vision_backbone=True)

    obs_dict = {
        "rgb1": torch.rand(8, 3, 224, 224),
        "rgb2": torch.rand(8, 3, 224, 224),
        "low_dim": torch.rand(8, 10),
    }

    encoded_features = encoder(obs_dict)
    print("Test 1 - Combined Inputs: Encoded Features Shape:", encoded_features.shape)
    assert encoded_features.shape[1] == 1034, "Unexpected feature dimension."

    # Test case with only RGB inputs
    shape_meta_rgb_only = {
        "rgb1": {"shape": (3, 224, 224), "type": "rgb"},
        "rgb2": {"shape": (3, 224, 224), "type": "rgb"},
    }

    encoder_rgb_only = ObsEncoder(
        shape_meta=shape_meta_rgb_only, share_vision_backbone=True
    )

    obs_dict_rgb_only = {
        "rgb1": torch.rand(8, 3, 224, 224),
        "rgb2": torch.rand(8, 3, 224, 224),
    }

    encoded_features_rgb_only = encoder_rgb_only(obs_dict_rgb_only)
    print("Test 2 - RGB Only: Encoded Features Shape:", encoded_features_rgb_only.shape)
    assert (
        encoded_features_rgb_only.shape[1] == 1024
    ), "Unexpected feature dimension for RGB only."

    print("All tests passed!")


if __name__ == "__main__":
    main()

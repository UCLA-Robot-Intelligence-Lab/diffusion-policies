import torch
import torch.nn as nn


class DinoV2FeatureExtractor(nn.Module):
    """Wrap DINOv2 backbone to return a flat feature vector per image."""

    def __init__(self, backbone: nn.Module, output: str = "cls", proj_dim: int = None):
        super().__init__()
        if output not in {"cls", "mean_pool"}:
            raise ValueError(f"Unsupported output type: {output}")

        self.backbone = backbone
        self.output = output
        self.proj = nn.Identity() if proj_dim is None else nn.LazyLinear(proj_dim)

    def _extract_from_dict(self, features: dict) -> torch.Tensor:
        if self.output == "cls":
            if "x_norm_clstoken" in features:
                return features["x_norm_clstoken"]
            if "cls_token" in features:
                return features["cls_token"]
            raise KeyError(
                "DINOv2 feature dict does not contain a cls token key. "
                "Expected one of: x_norm_clstoken, cls_token"
            )

        # self.output == "mean_pool"
        if "x_norm_patchtokens" in features:
            return features["x_norm_patchtokens"].mean(dim=1)
        if "patchtokens" in features:
            return features["patchtokens"].mean(dim=1)
        raise KeyError(
            "DINOv2 feature dict does not contain patch tokens for mean pooling. "
            "Expected one of: x_norm_patchtokens, patchtokens"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "forward_features"):
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)

        if isinstance(features, dict):
            feat = self._extract_from_dict(features)
        elif torch.is_tensor(features):
            if features.ndim == 2:
                feat = features
            elif features.ndim == 3:
                feat = features[:, 0] if self.output == "cls" else features.mean(dim=1)
            else:
                raise ValueError(f"Unsupported tensor feature shape: {tuple(features.shape)}")
        else:
            raise TypeError(f"Unsupported DINOv2 feature type: {type(features)}")

        if feat.ndim > 2:
            feat = feat.flatten(start_dim=1)

        return self.proj(feat)


def get_dinov2(
    name: str = "dinov2_vits14",
    pretrained: bool = True,
    output: str = "cls",
    proj_dim: int = None,
    repo: str = "facebookresearch/dinov2",
    force_reload: bool = False,
):
    """
    Load DINOv2 model via torch hub and expose flat image features.

    Args:
        name: Hub model name, e.g. dinov2_vits14 / dinov2_vitb14.
        pretrained: Load pretrained weights from hub.
        output: Feature aggregation mode: cls or mean_pool.
        proj_dim: Optional output projection dimension.
        repo: Torch hub repository.
        force_reload: Forwarded to torch.hub.load.
    """
    backbone = torch.hub.load(
        repo_or_dir=repo,
        model=name,
        pretrained=pretrained,
        force_reload=force_reload,
    )
    return DinoV2FeatureExtractor(backbone=backbone, output=output, proj_dim=proj_dim)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 as pretrained_resnet18, ResNet18_Weights
from spatial_softmax import SpatialSoftArgmax


class ResnetBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, stride=1, downsample=None):
        super(ResnetBlock, self).__init__()
        num_groups = out_channels // 16
        num_groups = max(num_groups, 1)

        self.conv1 = nn.Conv2d(
            inp_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x

        out = self.conv1(x)
        out = self.gn1(out)
        out = F.silu(out)

        out = self.conv2(out)
        out = self.gn2(out)

        out += identity
        out = F.silu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet18, self).__init__()
        self.inp_channels = 64

        # First convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=64)
        self.act = nn.SiLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # See spatial_softmax.py
        self.spatial_softmax = SpatialSoftArgmax(normalize=True)

        self.fc = nn.Linear(1024, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inp_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inp_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.GroupNorm(num_groups=out_channels // 16, num_channels=out_channels),
            )

        layers = []
        layers.append(block(self.inp_channels, out_channels, stride, downsample))
        self.inp_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.inp_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.act(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.spatial_softmax(x)

        x = self.fc(x)

        return x


def replace_bn_with_gn(model):
    # Replace BN2D with GN
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            setattr(
                model,
                name,
                nn.GroupNorm(
                    num_groups=max(num_channels // 16, 1), num_channels=num_channels
                ),
            )
        else:
            replace_bn_with_gn(child)


def get_resnet18(num_classes=1000, pretrained=False, weights_dir="./weights"):
    os.makedirs(weights_dir, exist_ok=True)

    if pretrained:
        weights_path = os.path.join(weights_dir, "resnet18.pth")
        if not os.path.exists(weights_path):
            print("Downloading pretrained weights...")
            weights = ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
            torch.save(weights, weights_path)
        else:
            print("Loading pretrained weights from local directory...")
            weights = torch.load(weights_path, weights_only=True)

        model = pretrained_resnet18(weights=None)
        model.load_state_dict(weights)

        replace_bn_with_gn(model)

        model.spatial_softmax = SpatialSoftArgmax(normalize=True)
        model.fc = nn.Linear(
            1024, num_classes
        )  # need to resize to handle spatial_softmax

        def forward(x):
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)

            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)

            x = model.spatial_softmax(x)  # Replace GAP
            x = model.fc(x)
            return x

        model.forward = forward
        return model

    else:
        # Trainable / editable model
        return ResNet18(ResnetBlock, [2, 2, 2, 2], num_classes=num_classes)


if __name__ == "__main__":
    model = get_resnet18(num_classes=10, pretrained=False)  # Non-pretrained version
    model.eval()

    # Dummy input: Batch size 2, 3 channels, 224x224 image
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    print("Output shape:", output.shape)  # Should be (2, 10)

    pretrained_model = get_resnet18(num_classes=10, pretrained=True)
    pretrained_model.eval()
    output_pretrained = pretrained_model(dummy_input)
    print("Pretrained Output shape:", output_pretrained.shape)  # Should be (2, 10)

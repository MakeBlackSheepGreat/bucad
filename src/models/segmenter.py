from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.runtime import optional_import, require_dependency


torch = optional_import("torch")
nn = optional_import("torch.nn")
smp = optional_import("segmentation_models_pytorch")


if nn is not None:
    class TinySegmentationNet(nn.Module):
        def __init__(self, in_channels: int = 3, classes: int = 1) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, classes, kernel_size=2, stride=2),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))
else:  # pragma: no cover - torch missing
    class TinySegmentationNet:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Torch is required to construct the segmenter.")


def create_segmenter(
    *,
    architecture: str = "unet",
    encoder_name: str = "resnet18",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
):
    require_dependency("torch", torch)
    require_dependency("torch.nn", nn)
    if architecture.lower() == "unet" and smp is not None:
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    return TinySegmentationNet(in_channels=in_channels, classes=classes)


def load_segmenter(
    model_config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    *,
    map_location: str = "cpu",
):
    model = create_segmenter(
        architecture=model_config.get("architecture", "unet"),
        encoder_name=model_config.get("encoder_name", "resnet18"),
        encoder_weights=model_config.get("encoder_weights", "imagenet"),
        in_channels=int(model_config.get("in_channels", 3)),
        classes=int(model_config.get("classes", 1)),
    )
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=map_location)
        state_dict = state.get("state_dict", state)
        model.load_state_dict(state_dict, strict=False)
    return model

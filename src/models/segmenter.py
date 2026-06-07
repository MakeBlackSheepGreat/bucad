"""Segmenter model factory and checkpoint loader helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.runtime import optional_import, require_dependency
from src.models.segmentation_components import CENetLite, TimmFeaturePyramidSegmenter


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
    **model_kwargs,
):
    require_dependency("torch", torch)
    require_dependency("torch.nn", nn)
    normalized_architecture = architecture.lower()
    if normalized_architecture in {"cenet_lite", "cenet-lite"}:
        return CENetLite(
            in_channels=in_channels,
            classes=classes,
            base_channels=int(model_kwargs.get("base_channels", 32)),
            use_dseb=bool(model_kwargs.get("use_dseb", True)),
            use_cfam=bool(model_kwargs.get("use_cfam", True)),
            use_nonlocal=bool(model_kwargs.get("use_nonlocal", True)),
            boundary_head=bool(model_kwargs.get("boundary_head", False)),
        )
    if normalized_architecture in {"cenet_pvtv2", "cenet-pvtv2", "pvtv2_cenet"}:
        return TimmFeaturePyramidSegmenter(
            encoder_name=encoder_name or "pvt_v2_b0",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            decoder_channels=int(model_kwargs.get("decoder_channels", 128)),
            use_dseb=bool(model_kwargs.get("use_dseb", True)),
            use_cfam=bool(model_kwargs.get("use_cfam", True)),
            use_nonlocal=bool(model_kwargs.get("use_nonlocal", True)),
            boundary_head=bool(model_kwargs.get("boundary_head", False)),
        )
    if smp is not None:
        smp_architectures = {
            "unet": smp.Unet,
            "unetplusplus": getattr(smp, "UnetPlusPlus", None),
            "unet++": getattr(smp, "UnetPlusPlus", None),
            "unet_plus_plus": getattr(smp, "UnetPlusPlus", None),
            "fpn": getattr(smp, "FPN", None),
            "deeplabv3": getattr(smp, "DeepLabV3", None),
            "deeplabv3plus": getattr(smp, "DeepLabV3Plus", None),
            "deeplabv3+": getattr(smp, "DeepLabV3Plus", None),
            "manet": getattr(smp, "MAnet", None),
            "linknet": getattr(smp, "Linknet", None),
            "pan": getattr(smp, "PAN", None),
            "pspnet": getattr(smp, "PSPNet", None),
        }
        model_factory = smp_architectures.get(normalized_architecture)
        if model_factory is not None:
            return model_factory(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
            )
    if normalized_architecture == "unet" and smp is not None:
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
    state = None
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=map_location)
        checkpoint_config = state.get("model_config") if isinstance(state, dict) else None
        if isinstance(checkpoint_config, dict):
            model_config = {**model_config, **checkpoint_config}
    extra_model_kwargs = {
        key: value
        for key, value in model_config.items()
        if key
        not in {
            "architecture",
            "encoder_name",
            "encoder_weights",
            "in_channels",
            "classes",
        }
    }
    model = create_segmenter(
        architecture=model_config.get("architecture", "unet"),
        encoder_name=model_config.get("encoder_name", "resnet18"),
        encoder_weights=model_config.get("encoder_weights", "imagenet"),
        in_channels=int(model_config.get("in_channels", 3)),
        classes=int(model_config.get("classes", 1)),
        **extra_model_kwargs,
    )
    if state is not None:
        state_dict = state.get("state_dict", state)
        model.load_state_dict(state_dict, strict=False)
    return model

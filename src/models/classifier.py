from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.runtime import optional_import, require_dependency


torch = optional_import("torch")
nn = optional_import("torch.nn")
F = optional_import("torch.nn.functional")
timm = optional_import("timm")


if nn is not None:
    class TinyCNNClassifier(nn.Module):
        def __init__(self, in_chans: int = 3, num_classes: int = 2) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_chans, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(64, num_classes)

        def forward(self, x):
            features = self.features(x)
            return self.classifier(features.flatten(1))
else:  # pragma: no cover - torch missing
    class TinyCNNClassifier:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Torch is required to construct the classifier.")


def create_classifier(
    model_name: str = "resnet18",
    *,
    pretrained: bool = True,
    in_chans: int = 3,
    num_classes: int = 2,
):
    require_dependency("torch", torch)
    require_dependency("torch.nn", nn)
    normalized_name = model_name.lower()
    if normalized_name in {"basic_cnn", "tiny_cnn"}:
        return TinyCNNClassifier(in_chans=in_chans, num_classes=num_classes)
    if timm is not None:
        return timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
        )
    return TinyCNNClassifier(in_chans=in_chans, num_classes=num_classes)


def load_classifier(
    model_config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    *,
    map_location: str = "cpu",
):
    model = create_classifier(
        model_name=model_config.get("name", "resnet18"),
        pretrained=bool(model_config.get("pretrained", False)),
        in_chans=int(model_config.get("in_chans", 3)),
        num_classes=int(model_config.get("num_classes", 2)),
    )
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=map_location)
        state_dict = state.get("state_dict", state.get("model", state))
        model.load_state_dict(state_dict, strict=False)
    return model


def classifier_probabilities(model, batch, *, device: str = "cpu"):
    require_dependency("torch", torch)
    require_dependency("torch.nn.functional", F)
    model = model.to(device)
    if batch.ndim == 3:
        batch = batch.unsqueeze(0)
    batch = batch.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs


def resolve_gradcam_target_layer(model) -> Any | None:
    for candidate in ("layer4", "features", "blocks", "conv_head", "backbone"):
        layer = getattr(model, candidate, None)
        if layer is not None:
            if candidate == "backbone":
                nested = resolve_gradcam_target_layer(layer)
                if nested is not None:
                    return nested
            try:
                return layer[-1] if hasattr(layer, "__getitem__") and len(layer) > 0 else layer
            except TypeError:
                return layer
    return None

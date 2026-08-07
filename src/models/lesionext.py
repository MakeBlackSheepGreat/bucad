"""LesioNeXt-BUS: uncertainty-conditioned global-local lesion diagnosis."""

from __future__ import annotations

from typing import Any

from src.utils.runtime import optional_import


torch = optional_import("torch")
nn = optional_import("torch.nn")
F = optional_import("torch.nn.functional")


if nn is not None:
    class ConvNormAct(nn.Module):
        """Compact convolutional unit used by the LesioNeXt encoder and decoder."""

        def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )

        def forward(self, x):
            return self.block(x)


    class BoundaryFrequencyBlock(nn.Module):
        """Separate smooth context from high-frequency edge evidence."""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.context = ConvNormAct(channels, channels)
            self.edge = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
            )
            self.gate = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1),
                nn.GELU(),
                nn.Conv2d(channels, channels, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            smooth = self.context(x)
            high_frequency = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            edge = self.edge(high_frequency)
            gate = self.gate(torch.cat([smooth, edge], dim=1))
            return x + smooth + gate * edge


    class DecoderBlock(nn.Module):
        """Upsample one stage and fuse the matching encoder feature."""

        def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
            super().__init__()
            self.fuse = ConvNormAct(in_channels + skip_channels, out_channels)
            self.refine = BoundaryFrequencyBlock(out_channels)

        def forward(self, x, skip):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return self.refine(self.fuse(torch.cat([x, skip], dim=1)))


    class LesioNeXt(nn.Module):
        """Unified segmentation-classification model for breast ultrasound images."""

        input_mode = "single_image"

        def __init__(
            self,
            *,
            in_channels: int = 3,
            num_classes: int = 2,
            classes: int = 1,
            widths: tuple[int, ...] | list[int] = (32, 64, 128, 192),
            embedding_dim: int = 128,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            if len(widths) != 4:
                raise ValueError("LesioNeXt expects exactly four encoder widths.")
            self.widths = tuple(int(width) for width in widths)
            self.num_classes = int(num_classes)
            self.classes = int(classes)
            self.stem = ConvNormAct(in_channels, self.widths[0], stride=1)
            self.enc1 = nn.Sequential(
                BoundaryFrequencyBlock(self.widths[0]),
                ConvNormAct(self.widths[0], self.widths[1], stride=2),
            )
            self.enc2 = nn.Sequential(
                BoundaryFrequencyBlock(self.widths[1]),
                ConvNormAct(self.widths[1], self.widths[2], stride=2),
            )
            self.enc3 = nn.Sequential(
                BoundaryFrequencyBlock(self.widths[2]),
                ConvNormAct(self.widths[2], self.widths[3], stride=2),
                BoundaryFrequencyBlock(self.widths[3]),
            )
            self.decoder3 = DecoderBlock(self.widths[3], self.widths[2], self.widths[2])
            self.decoder2 = DecoderBlock(self.widths[2], self.widths[1], self.widths[1])
            self.decoder1 = DecoderBlock(self.widths[1], self.widths[0], self.widths[0])
            self.mask_head = nn.Conv2d(self.widths[0], self.classes, 1)
            self.boundary_head = nn.Conv2d(self.widths[0], self.classes, 1)
            self.uncertainty_head = nn.Conv2d(self.widths[0], self.classes, 1)
            self.global_projection = nn.Sequential(
                nn.Linear(self.widths[3], embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
            )
            self.local_projection = nn.Sequential(
                nn.Linear(self.widths[0], embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
            )
            self.router = nn.Sequential(
                nn.LayerNorm(embedding_dim * 2 + 2),
                nn.Linear(embedding_dim * 2 + 2, embedding_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(embedding_dim, 1),
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(embedding_dim * 2),
                nn.Dropout(float(dropout)),
                nn.Linear(embedding_dim * 2, self.num_classes),
            )
            self.gradcam_layer = self.decoder1.refine.edge

        @staticmethod
        def _weighted_pool(features, weights):
            weights = F.interpolate(weights, size=features.shape[-2:], mode="bilinear", align_corners=False)
            numerator = (features * weights).sum(dim=(2, 3))
            denominator = weights.sum(dim=(2, 3)).clamp_min(1e-6)
            return numerator / denominator

        def forward_with_aux(self, x) -> dict[str, Any]:
            stem = self.stem(x)
            skip1 = stem
            skip2 = self.enc1(stem)
            skip3 = self.enc2(skip2)
            bottleneck = self.enc3(skip3)
            decoded = self.decoder3(bottleneck, skip3)
            decoded = self.decoder2(decoded, skip2)
            decoded = self.decoder1(decoded, skip1)
            decoded = F.interpolate(decoded, size=x.shape[-2:], mode="bilinear", align_corners=False)
            mask_logits = self.mask_head(decoded)
            boundary_logits = self.boundary_head(decoded)
            uncertainty_logits = self.uncertainty_head(decoded)
            mask_probability = torch.sigmoid(mask_logits)
            uncertainty_probability = torch.sigmoid(uncertainty_logits)
            reliable_roi = mask_probability * (1.0 - uncertainty_probability)
            global_embedding = self.global_projection(F.adaptive_avg_pool2d(bottleneck, 1).flatten(1))
            local_embedding = self.local_projection(self._weighted_pool(decoded, reliable_roi))
            area_ratio = mask_probability.mean(dim=(1, 2, 3)).unsqueeze(1)
            uncertainty_mean = uncertainty_probability.mean(dim=(1, 2, 3)).unsqueeze(1)
            router_input = torch.cat([global_embedding, local_embedding, area_ratio, uncertainty_mean], dim=1)
            roi_weight = torch.sigmoid(self.router(router_input))
            fused_local = local_embedding * roi_weight
            class_logits = self.classifier(torch.cat([global_embedding, fused_local], dim=1))
            return {
                "mask": mask_logits,
                "boundary": boundary_logits,
                "uncertainty": uncertainty_logits,
                "class_logits": class_logits,
                "roi_weight": roi_weight,
                "area_ratio": area_ratio,
                "uncertainty_mean": uncertainty_mean,
                "global_embedding": global_embedding,
                "local_embedding": local_embedding,
            }

        def forward(self, x):
            return self.forward_with_aux(x)

        def get_gradcam_target_layer(self):
            return self.gradcam_layer


    class LesioNeXtClassifier(nn.Module):
        """Classifier-compatible wrapper around the unified LesioNeXt model."""

        input_mode = "single_image"

        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.network = LesioNeXt(**kwargs)
            self.gradcam_layer = self.network.gradcam_layer

        def forward_with_embedding(self, image_full=None, image_roi=None, **batch):
            image = image_full if image_full is not None else batch.get("image")
            if image is None:
                raise ValueError("LesioNeXtClassifier requires image_full or image.")
            outputs = self.network(image)
            return outputs["class_logits"], outputs

        def forward(self, image_full=None, image_roi=None, **batch):
            logits, _ = self.forward_with_embedding(image_full=image_full, image_roi=image_roi, **batch)
            return logits

        def get_gradcam_target_layer(self):
            return self.network.get_gradcam_target_layer()
else:  # pragma: no cover - torch missing
    class LesioNeXt:  # type: ignore[override]
        """Placeholder used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Torch is required to construct LesioNeXt.")

    class LesioNeXtClassifier:  # type: ignore[override]
        """Placeholder used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Torch is required to construct LesioNeXtClassifier.")

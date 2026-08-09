"""Lesion-evidence aligned ConvNeXt classifier for BUSBRA training."""

from __future__ import annotations

import math
from typing import Any

from src.utils.runtime import optional_import, require_dependency


torch = optional_import("torch")
nn = optional_import("torch.nn")
F = optional_import("torch.nn.functional")


if nn is not None:
    class LesionEvidenceConvNeXtClassifier(nn.Module):
        """Single ConvNeXt classifier with weak BBOX-guided evidence pooling."""

        input_mode = "single_image"

        def __init__(
            self,
            *,
            backbone_name: str = "convnext_tiny",
            pretrained: bool = True,
            in_chans: int = 3,
            num_classes: int = 2,
            evidence_alpha_init: float = 0.02,
            evidence_gate_mode: str = "static",
            evidence_gate_gain_init: float = 0.0,
            local_evidence_auxiliary: bool = False,
            evidence_multires: bool = False,
            auxiliary_evidence_stage_index: int = 2,
            head_dropout: float = 0.2,
        ) -> None:
            """Build a ConvNeXt backbone with one lightweight lesion evidence head."""
            super().__init__()
            timm = optional_import("timm")
            require_dependency("timm", timm)
            self.backbone_name = str(backbone_name)
            self.backbone = timm.create_model(
                self.backbone_name,
                pretrained=bool(pretrained),
                in_chans=int(in_chans),
                num_classes=0,
                global_pool="",
            )
            feature_dim = int(getattr(self.backbone, "num_features", 0))
            if feature_dim <= 0:
                raise RuntimeError("ConvNeXt backbone did not expose num_features.")
            self.feature_dim = feature_dim
            self.evidence_head = nn.Conv2d(feature_dim, 1, kernel_size=1, bias=True)
            nn.init.zeros_(self.evidence_head.weight)
            nn.init.zeros_(self.evidence_head.bias)
            self.evidence_multires = bool(evidence_multires)
            self.auxiliary_evidence_stage_index = int(auxiliary_evidence_stage_index)
            self.auxiliary_evidence_head = None
            if self.evidence_multires:
                feature_info = getattr(self.backbone, "feature_info", None)
                if feature_info is None:
                    channels = []
                elif hasattr(feature_info, "channels"):
                    channels = list(feature_info.channels())
                else:
                    channels = [int(item["num_chs"]) for item in feature_info]
                if not 0 <= self.auxiliary_evidence_stage_index < len(channels):
                    raise ValueError(
                        "auxiliary_evidence_stage_index must reference a ConvNeXt feature stage."
                    )
                auxiliary_dim = int(channels[self.auxiliary_evidence_stage_index])
                self.auxiliary_evidence_head = nn.Conv2d(
                    auxiliary_dim,
                    1,
                    kernel_size=1,
                    bias=True,
                )
                nn.init.zeros_(self.auxiliary_evidence_head.weight)
                nn.init.zeros_(self.auxiliary_evidence_head.bias)
            normalized_gate_mode = str(evidence_gate_mode).strip().lower()
            if normalized_gate_mode not in {"disabled", "static", "confidence"}:
                raise ValueError("evidence_gate_mode must be 'disabled', 'static', or 'confidence'.")
            self.evidence_gate_mode = normalized_gate_mode
            self.local_evidence_auxiliary = bool(local_evidence_auxiliary)
            alpha = min(max(float(evidence_alpha_init), 1e-4), 0.5)
            self.evidence_alpha_logit = nn.Parameter(
                torch.tensor(float(torch.logit(torch.tensor(alpha))))
            )
            self.evidence_gate_gain = nn.Parameter(torch.tensor(float(evidence_gate_gain_init)))
            self.head_norm = nn.LayerNorm(feature_dim)
            self.head_dropout = nn.Dropout(float(head_dropout))
            self.classifier = nn.Linear(feature_dim, int(num_classes))
            self.gradcam_layer = self.backbone.stages[-1].blocks[-1]
            self.last_evidence_map: Any | None = None
            self.last_evidence_maps: list[Any] = []
            self.last_evidence_alpha: Any | None = None
            self.last_evidence_concentration: Any | None = None
            self.last_evidence_agreement: Any | None = None
            self.last_local_evidence_logits: Any | None = None
            self.last_alignment_error_weight: Any | None = None
            self.last_logits: Any | None = None

        def _features(self, image):
            """Return final ConvNeXt features and optional stage-level evidence features."""
            if not self.evidence_multires:
                return self.backbone.forward_features(image), None
            features = self.backbone.stem(image)
            auxiliary_features = None
            for stage_index, stage in enumerate(self.backbone.stages):
                features = stage(features)
                if stage_index == self.auxiliary_evidence_stage_index:
                    auxiliary_features = features
            norm_pre = getattr(self.backbone, "norm_pre", None)
            if norm_pre is not None:
                features = norm_pre(features)
            if auxiliary_features is None:
                raise RuntimeError("Configured auxiliary evidence stage was not reached.")
            return features, auxiliary_features

        def _evidence_alpha(self, evidence_weights, local_embedding, global_embedding):
            """Return one evidence interpolation weight per sample plus gate diagnostics."""
            batch_size, _, height, width = evidence_weights.shape
            base_logit = self.evidence_alpha_logit.to(dtype=global_embedding.dtype)
            if self.evidence_gate_mode == "disabled":
                alpha = torch.zeros(batch_size, device=global_embedding.device, dtype=global_embedding.dtype)
                concentration = torch.zeros_like(alpha)
                agreement = torch.ones_like(alpha)
                return alpha, concentration, agreement
            if self.evidence_gate_mode == "static":
                alpha = torch.sigmoid(base_logit).expand(batch_size)
                concentration = torch.zeros_like(alpha)
                agreement = torch.ones_like(alpha)
                return alpha, concentration, agreement

            flat_weights = evidence_weights.flatten(1).clamp_min(1e-8)
            spatial_entropy = -(flat_weights * flat_weights.log()).sum(dim=1)
            max_entropy = math.log(max(2, int(height * width)))
            concentration = (1.0 - spatial_entropy / max_entropy).clamp(0.0, 1.0)
            agreement = (
                0.5
                * (
                    F.cosine_similarity(local_embedding, global_embedding, dim=1, eps=1e-6)
                    + 1.0
                )
            ).clamp(0.0, 1.0)
            confidence = 0.5 * (concentration + agreement)
            gain = self.evidence_gate_gain.to(dtype=global_embedding.dtype)
            alpha = torch.sigmoid(base_logit + gain * (confidence - 0.5))
            return alpha, concentration, agreement

        def forward_with_embedding(self, image=None, image_full=None, bbox=None, **_kwargs):
            """Return logits and evidence-aware embedding while accepting training-only BBOX data."""
            del bbox
            tensor = image_full if image_full is not None else image
            if tensor is None:
                raise ValueError("LesionEvidenceConvNeXtClassifier requires an image tensor.")
            features, auxiliary_features = self._features(tensor)
            evidence_logits = self.evidence_head(features)
            evidence_maps = [evidence_logits]
            if self.auxiliary_evidence_head is not None and auxiliary_features is not None:
                evidence_maps.insert(0, self.auxiliary_evidence_head(auxiliary_features))
            evidence_weights = torch.softmax(evidence_logits.flatten(1), dim=1).reshape_as(evidence_logits)
            global_embedding = F.adaptive_avg_pool2d(features, 1).flatten(1)
            local_embedding = (features * evidence_weights).sum(dim=(2, 3))
            alpha, concentration, agreement = self._evidence_alpha(
                evidence_weights,
                local_embedding,
                global_embedding,
            )
            embedding = global_embedding + alpha.unsqueeze(1) * (local_embedding - global_embedding)
            logits = self.classifier(self.head_dropout(self.head_norm(embedding)))
            if self.training and self.local_evidence_auxiliary:
                self.last_local_evidence_logits = self.classifier(
                    self.head_dropout(self.head_norm(local_embedding))
                )
            else:
                self.last_local_evidence_logits = None
            self.last_evidence_map = evidence_logits
            self.last_evidence_maps = evidence_maps
            self.last_logits = logits
            self.last_evidence_alpha = alpha.detach()
            self.last_evidence_concentration = concentration.detach()
            self.last_evidence_agreement = agreement.detach()
            return logits, embedding

        def forward(self, image=None, image_full=None, bbox=None, **kwargs):
            """Return malignant/benign logits for one image batch."""
            logits, _embedding = self.forward_with_embedding(
                image=image,
                image_full=image_full,
                bbox=bbox,
                **kwargs,
            )
            return logits

        def get_gradcam_target_layer(self):
            """Return the final ConvNeXt block used by explanation tooling."""
            return self.gradcam_layer

else:
    class LesionEvidenceConvNeXtClassifier:  # type: ignore[override]
        """Dependency placeholder used when PyTorch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct LesionEvidenceConvNeXtClassifier.")

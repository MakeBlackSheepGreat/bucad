"""Classifier model factory, checkpoint loader, and probability helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.models.attnres import DepthwiseBlockAttnResConvNeXtClassifier
from src.models.lens import LesionEvidenceConvNeXtClassifier
from src.models.dara import (
    CrossStageDeltaHistoryFusion,
    ReliabilityGatedSharedRoutedExpertHead,
    SharedRoutedExpertHead,
    StageAttentionResidualFusion,
)
from src.utils.runtime import optional_import, require_dependency


torch = optional_import("torch")
nn = optional_import("torch.nn")
F = optional_import("torch.nn.functional")
timm = optional_import("timm")
torchvision_models = optional_import("torchvision.models")


if nn is not None:
    VALID_LESIONEXT_STAGE_INDEX_GROUPS = {
        (2, 3, 4),
        (3, 4),
        (4,),
    }

    class TinyCNNClassifier(nn.Module):
        """Small CNN fallback used for smoke tests and missing heavy dependencies."""

        def __init__(self, in_chans: int = 3, num_classes: int = 2) -> None:
            """Build a compact three-block convolutional classifier."""
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
            """Return class logits for one image batch."""
            features = self.features(x)
            return self.classifier(features.flatten(1))


    class GeMPool2d(nn.Module):
        """Generalized mean pooling with an optional learnable exponent."""

        def __init__(self, p: float = 3.0, eps: float = 1e-6, *, learnable: bool = True) -> None:
            """Initialize GeM pooling with a stable positive exponent."""
            super().__init__()
            if torch is None:
                raise RuntimeError("Torch is required to construct GeMPool2d.")
            initial = torch.tensor(float(p), dtype=torch.float32)
            self.p = nn.Parameter(initial) if learnable else initial
            self.eps = float(eps)

        def forward(self, x):
            """Pool one feature map batch into one vector per sample."""
            exponent = self.p.clamp(min=self.eps) if hasattr(self.p, "clamp") else self.p
            pooled = F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(exponent), 1)
            return pooled.pow(exponent.reciprocal() if hasattr(exponent, "reciprocal") else 1.0 / exponent)


    class ArcMarginProduct(nn.Module):
        """Cosine-margin classification head inspired by ArcFace-style large-margin learning."""

        def __init__(
            self,
            in_features: int,
            out_features: int,
            *,
            scale: float = 16.0,
            margin: float = 0.15,
            easy_margin: bool = False,
        ) -> None:
            """Build a normalized linear head with additive angular margin."""
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.scale = float(scale)
            self.margin = float(margin)
            self.easy_margin = bool(easy_margin)
            self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
            nn.init.xavier_uniform_(self.weight)
            self.cos_m = math.cos(self.margin)
            self.sin_m = math.sin(self.margin)
            self.th = math.cos(math.pi - self.margin)
            self.mm = math.sin(math.pi - self.margin) * self.margin

        def forward(self, embeddings, labels=None):
            """Return scaled cosine logits, applying angular margin only when labels are provided."""
            normalized_embeddings = F.normalize(embeddings, dim=1)
            normalized_weight = F.normalize(self.weight, dim=1)
            cosine = F.linear(normalized_embeddings, normalized_weight).clamp(-1.0, 1.0)
            if labels is None:
                return cosine * self.scale
            sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(1e-6))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
            logits = one_hot * phi + (1.0 - one_hot) * cosine
            return logits * self.scale


    class MixStyle2d(nn.Module):
        """Training-time feature-statistics mixing for lightweight domain generalization."""

        def __init__(self, *, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6) -> None:
            """Initialize MixStyle with application probability and Beta mixing strength."""
            super().__init__()
            self.p = float(p)
            self.alpha = float(alpha)
            self.eps = float(eps)

        def forward(self, x):
            """Mix instance-level feature statistics across samples during training only."""
            if (
                not self.training
                or self.p <= 0.0
                or self.alpha <= 0.0
                or x.ndim != 4
                or x.shape[0] < 2
                or torch.rand(1, device=x.device).item() > self.p
            ):
                return x
            batch_size = x.shape[0]
            mu = x.mean(dim=(2, 3), keepdim=True)
            var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
            sigma = (var + self.eps).sqrt()
            normalized = (x - mu) / sigma
            perm = torch.randperm(batch_size, device=x.device)
            beta = torch.distributions.Beta(self.alpha, self.alpha)
            lam = beta.sample((batch_size, 1, 1, 1)).to(device=x.device, dtype=x.dtype)
            mixed_mu = lam * mu + (1.0 - lam) * mu[perm]
            mixed_sigma = lam * sigma + (1.0 - lam) * sigma[perm]
            return normalized * mixed_sigma + mixed_mu


    class LightweightSelfAttention(nn.Module):
        """Single-layer MHSA block used for low-cost global context modeling."""

        def __init__(
            self,
            dim: int,
            *,
            heads: int = 4,
            dropout: float = 0.1,
            mlp_ratio: float = 2.0,
        ) -> None:
            """Build one residual attention block over flattened spatial tokens."""
            super().__init__()
            hidden_dim = max(dim, int(round(dim * mlp_ratio)))
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
            )

        def forward(self, tokens):
            """Refine token features with one attention block and MLP."""
            normalized = self.norm1(tokens)
            attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
            tokens = tokens + attended
            return tokens + self.mlp(self.norm2(tokens))


    class GlobalResponseNorm2d(nn.Module):
        """ConvNeXt V2-style global response normalization for channels-first tensors."""

        def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
            """Initialize learnable response calibration for one channel dimension."""
            super().__init__()
            self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
            self.eps = float(eps)

        def forward(self, x):
            """Calibrate responses using global channel competition statistics."""
            response = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
            normalized = response / (response.mean(dim=1, keepdim=True) + self.eps)
            return x + self.gamma * (x * normalized) + self.beta


    class EfficientChannelAttention2d(nn.Module):
        """ECA-Net style lightweight channel attention for channels-first tensors."""

        def __init__(self, channels: int, *, kernel_size: int | None = None) -> None:
            """Initialize one local cross-channel interaction block."""
            super().__init__()
            resolved_kernel = self._resolve_kernel_size(int(channels), kernel_size)
            padding = (resolved_kernel - 1) // 2
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv = nn.Conv1d(1, 1, kernel_size=resolved_kernel, padding=padding, bias=False)
            self.activation = nn.Sigmoid()
            self.kernel_size = resolved_kernel

        @staticmethod
        def _resolve_kernel_size(channels: int, kernel_size: int | None) -> int:
            """Return an odd kernel size following the ECA paper heuristic."""
            if kernel_size is not None:
                resolved = max(1, int(kernel_size))
            else:
                resolved = int(abs((torch.log2(torch.tensor(float(channels))).item() / 2.0) + 0.5))
            if resolved % 2 == 0:
                resolved += 1
            return max(1, resolved)

        def forward(self, x):
            """Recalibrate channels with local neighborhood interaction."""
            pooled = self.avg_pool(x).flatten(2).transpose(1, 2)
            weights = self.activation(self.conv(pooled)).transpose(1, 2).unsqueeze(-1)
            return x * weights.expand_as(x)


    class SpectralChannelGate2d(nn.Module):
        """Frequency-aware per-channel gate using low-frequency FFT magnitude summaries."""

        def __init__(
            self,
            channels: int,
            *,
            freq_size: int = 4,
            hidden_dim: int | None = None,
            include_spatial: bool = True,
        ) -> None:
            """Build a shared per-channel MLP over spectral and optional spatial descriptors."""
            super().__init__()
            if torch is None or not hasattr(torch, "fft") or not hasattr(torch.fft, "rfft2"):
                raise RuntimeError("torch.fft.rfft2 is required to construct SpectralChannelGate2d.")
            self.channels = int(channels)
            self.freq_size = max(1, int(freq_size))
            self.include_spatial = bool(include_spatial)
            input_dim = self.freq_size * self.freq_size + (1 if self.include_spatial else 0)
            resolved_hidden_dim = int(hidden_dim or max(16, input_dim * 2))
            self.norm = nn.LayerNorm(input_dim)
            self.fc1 = nn.Linear(input_dim, resolved_hidden_dim)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(resolved_hidden_dim, 1)
            self.last_weights = None

        def _spectral_descriptor(self, x):
            """Summarize one feature map batch with a fixed low-frequency magnitude grid."""
            freq = torch.fft.rfft2(x.float(), norm="ortho")
            magnitude = freq.abs()
            pooled_height = min(self.freq_size, int(magnitude.shape[-2]))
            pooled_width = min(self.freq_size, int(magnitude.shape[-1]))
            pooled = F.adaptive_avg_pool2d(magnitude, output_size=(pooled_height, pooled_width))
            pad_height = self.freq_size - pooled_height
            pad_width = self.freq_size - pooled_width
            if pad_height > 0 or pad_width > 0:
                pooled = F.pad(pooled, (0, pad_width, 0, pad_height))
            return pooled.flatten(2).to(dtype=x.dtype)

        def forward(self, x):
            """Reweight channels with a compact spectral gate."""
            spectral = self._spectral_descriptor(x)
            if self.include_spatial:
                spatial = x.mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
                descriptor = torch.cat([spectral, spatial], dim=2)
            else:
                descriptor = spectral
            weights = torch.sigmoid(self.fc2(self.act(self.fc1(self.norm(descriptor))))).squeeze(-1)
            self.last_weights = weights.detach()
            return x * weights.unsqueeze(-1).unsqueeze(-1), weights


    class CoordinateAttention2d(nn.Module):
        """Coordinate attention with directional global encoding and location awareness."""

        def __init__(
            self,
            channels: int,
            *,
            reduction: int = 32,
            min_channels: int = 8,
        ) -> None:
            """Build one lightweight coordinate attention block for 2D feature maps."""
            super().__init__()
            reduced_channels = max(int(min_channels), int(channels) // max(1, int(reduction)))
            self.reduce = nn.Sequential(
                nn.Conv2d(int(channels), reduced_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduced_channels),
                nn.Hardswish(),
            )
            self.expand_h = nn.Conv2d(reduced_channels, int(channels), kernel_size=1, bias=False)
            self.expand_w = nn.Conv2d(reduced_channels, int(channels), kernel_size=1, bias=False)

        def forward(self, x):
            """Apply direction-aware attention maps without collapsing positional structure."""
            _, _, height, width = x.shape
            pooled_h = x.mean(dim=3, keepdim=True)
            pooled_w = x.mean(dim=2, keepdim=True).transpose(2, 3)
            encoded = self.reduce(torch.cat([pooled_h, pooled_w], dim=2))
            attention_h, attention_w = torch.split(encoded, [height, width], dim=2)
            attention_w = attention_w.transpose(2, 3)
            return x * self.expand_h(attention_h).sigmoid() * self.expand_w(attention_w).sigmoid()


    class GlobalContextBlock2d(nn.Module):
        """GCNet-style lightweight global context block for channels-first feature maps."""

        def __init__(
            self,
            channels: int,
            *,
            reduction: int = 16,
            min_channels: int = 8,
        ) -> None:
            """Build one content-adaptive global context aggregation block."""
            super().__init__()
            reduced_channels = max(int(min_channels), int(channels) // max(1, int(reduction)))
            self.context_mask = nn.Conv2d(int(channels), 1, kernel_size=1)
            self.transform = nn.Sequential(
                nn.Conv2d(int(channels), reduced_channels, kernel_size=1, bias=False),
                nn.LayerNorm([reduced_channels, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduced_channels, int(channels), kernel_size=1, bias=False),
            )

        def forward(self, x):
            """Add one query-independent global context residual to the input feature map."""
            batch_size, channels, height, width = x.shape
            attention = self.context_mask(x).reshape(batch_size, 1, height * width)
            attention = torch.softmax(attention, dim=2)
            context = torch.matmul(x.reshape(batch_size, channels, height * width), attention.transpose(1, 2))
            context = context.reshape(batch_size, channels, 1, 1)
            return x + self.transform(context)


    class LargeKernelAttention2d(nn.Module):
        """A lightweight VAN-style large kernel attention block for 2D features."""

        def __init__(self, channels: int, *, kernel_size: int = 7, dilation: int = 3) -> None:
            """Build a depthwise large-kernel spatial attention module."""
            super().__init__()
            local_kernel = max(3, int(kernel_size))
            if local_kernel % 2 == 0:
                local_kernel += 1
            dilated_kernel = max(3, local_kernel - 2)
            if dilated_kernel % 2 == 0:
                dilated_kernel += 1
            dilated_padding = ((dilated_kernel - 1) // 2) * int(dilation)
            self.pre = nn.Conv2d(int(channels), int(channels), kernel_size=1, bias=False)
            self.depthwise_local = nn.Conv2d(
                int(channels),
                int(channels),
                kernel_size=local_kernel,
                padding=local_kernel // 2,
                groups=int(channels),
                bias=False,
            )
            self.depthwise_dilated = nn.Conv2d(
                int(channels),
                int(channels),
                kernel_size=dilated_kernel,
                padding=dilated_padding,
                dilation=int(dilation),
                groups=int(channels),
                bias=False,
            )
            self.pointwise = nn.Conv2d(int(channels), int(channels), kernel_size=1, bias=False)

        def forward(self, x):
            """Apply large-kernel spatial attention as a residual modulation."""
            residual = x
            attention = self.pre(x)
            attention = self.depthwise_local(attention)
            attention = self.depthwise_dilated(attention)
            attention = self.pointwise(attention).sigmoid()
            return residual + residual * attention


    class SelectiveReceptiveFieldMixer2d(nn.Module):
        """Selective receptive-field mixer inspired by SKNet and InceptionNeXt."""

        def __init__(
            self,
            channels: int,
            *,
            square_kernel: int = 3,
            band_kernel_sizes: list[int] | tuple[int, ...] | None = None,
            reduction: int = 16,
            min_channels: int = 32,
            use_identity: bool = True,
        ) -> None:
            """Build one lightweight dynamic local mixer with square and band branches."""
            super().__init__()
            self.channels = int(channels)
            self.square_kernel = self._normalize_kernel(square_kernel)
            self.band_kernel_sizes = self._normalize_band_kernels(band_kernel_sizes)
            self.use_identity = bool(use_identity)
            self.branches = nn.ModuleList()
            if self.use_identity:
                self.branches.append(nn.Identity())
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.channels,
                        self.channels,
                        kernel_size=self.square_kernel,
                        padding=self.square_kernel // 2,
                        groups=self.channels,
                        bias=False,
                    ),
                    nn.GELU(),
                )
            )
            for kernel_size in self.band_kernel_sizes:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(
                            self.channels,
                            self.channels,
                            kernel_size=(1, kernel_size),
                            padding=(0, kernel_size // 2),
                            groups=self.channels,
                            bias=False,
                        ),
                        nn.Conv2d(
                            self.channels,
                            self.channels,
                            kernel_size=(kernel_size, 1),
                            padding=(kernel_size // 2, 0),
                            groups=self.channels,
                            bias=False,
                        ),
                        nn.GELU(),
                    )
                )
            hidden_channels = max(int(min_channels), self.channels // max(1, int(reduction)))
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(self.channels, hidden_channels, kernel_size=1, bias=False)
            self.act = nn.GELU()
            self.fc2 = nn.Conv2d(
                hidden_channels,
                len(self.branches) * self.channels,
                kernel_size=1,
                bias=False,
            )
            self.num_branches = len(self.branches)

        @staticmethod
        def _normalize_kernel(kernel_size: int) -> int:
            """Return one positive odd kernel size."""
            resolved = max(1, int(kernel_size))
            if resolved % 2 == 0:
                resolved += 1
            return resolved

        @classmethod
        def _normalize_band_kernels(
            cls,
            band_kernel_sizes: list[int] | tuple[int, ...] | None,
        ) -> tuple[int, ...]:
            """Return validated odd band kernels for anisotropic local branches."""
            if band_kernel_sizes is None:
                values = (7, 11)
            else:
                values = tuple(cls._normalize_kernel(value) for value in band_kernel_sizes)
            if len(values) == 0:
                raise ValueError("band_kernel_sizes must contain at least one kernel size.")
            return values

        def forward(self, x):
            """Fuse multiple local receptive fields with channel-wise soft selection."""
            branch_outputs = [branch(x) for branch in self.branches]
            stacked = torch.stack(branch_outputs, dim=1)
            fused = stacked.sum(dim=1)
            descriptor = self.avg_pool(fused)
            logits = self.fc2(self.act(self.fc1(descriptor)))
            logits = logits.view(x.shape[0], self.num_branches, self.channels, 1, 1)
            weights = torch.softmax(logits, dim=1)
            return (stacked * weights).sum(dim=1)


    class SelectiveScaleGate(nn.Module):
        """Sample-adaptive scale gate inspired by selective kernel branch selection."""

        def __init__(
            self,
            scale_dims: list[int] | tuple[int, ...],
            *,
            hidden_dim: int | None = None,
            temperature: float = 1.0,
        ) -> None:
            """Build a light gating MLP over concatenated multi-scale embeddings."""
            super().__init__()
            self.scale_dims = tuple(int(dim) for dim in scale_dims)
            self.num_scales = len(self.scale_dims)
            fused_dim = int(sum(self.scale_dims))
            hidden = int(hidden_dim or max(128, fused_dim // max(4, self.num_scales * 2)))
            self.norm = nn.LayerNorm(fused_dim)
            self.fc1 = nn.Linear(fused_dim, hidden)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden, self.num_scales)
            self.temperature = max(float(temperature), 1e-4)

        def forward(self, embeddings: list[Any]) -> tuple[list[Any], Any]:
            """Reweight each active scale while keeping the average scale near one."""
            if len(embeddings) != self.num_scales:
                raise ValueError(
                    f"SelectiveScaleGate expected {self.num_scales} embeddings, got {len(embeddings)}."
                )
            fused = torch.cat(embeddings, dim=1)
            logits = self.fc2(self.act(self.fc1(self.norm(fused)))) / self.temperature
            weights = torch.softmax(logits, dim=1) * self.num_scales
            gated = [embedding * weights[:, index : index + 1] for index, embedding in enumerate(embeddings)]
            return gated, weights


    class SemanticScaleGate(nn.Module):
        """Use the deepest semantic descriptor to modulate active multi-scale embeddings."""

        def __init__(
            self,
            source_dim: int,
            scale_dims: list[int] | tuple[int, ...],
            *,
            hidden_dim: int | None = None,
            temperature: float = 1.0,
        ) -> None:
            """Build one stage4-conditioned scale gate over active branch embeddings."""
            super().__init__()
            self.source_dim = int(source_dim)
            self.scale_dims = tuple(int(dim) for dim in scale_dims)
            self.num_scales = len(self.scale_dims)
            hidden = int(hidden_dim or max(128, self.source_dim // 2))
            self.norm = nn.LayerNorm(self.source_dim)
            self.fc1 = nn.Linear(self.source_dim, hidden)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden, self.num_scales)
            self.temperature = max(float(temperature), 1e-4)

        def forward(self, embeddings: list[Any], source_embedding) -> tuple[list[Any], Any]:
            """Predict per-scale weights from one deepest semantic embedding."""
            if len(embeddings) != self.num_scales:
                raise ValueError(
                    f"SemanticScaleGate expected {self.num_scales} embeddings, got {len(embeddings)}."
                )
            logits = self.fc2(self.act(self.fc1(self.norm(source_embedding)))) / self.temperature
            weights = torch.softmax(logits, dim=1) * self.num_scales
            gated = [embedding * weights[:, index : index + 1] for index, embedding in enumerate(embeddings)]
            return gated, weights


    class ResidualSemanticScaleGate(nn.Module):
        """Conservative stage4-conditioned scale gate with identity-centered residual weights."""

        def __init__(
            self,
            source_dim: int,
            scale_dims: list[int] | tuple[int, ...],
            *,
            hidden_dim: int | None = None,
            delta: float = 0.15,
            anchor_last_scale: bool = True,
        ) -> None:
            """Build one residual semantic gate that only nudges scale weights around one."""
            super().__init__()
            self.source_dim = int(source_dim)
            self.scale_dims = tuple(int(dim) for dim in scale_dims)
            self.num_scales = len(self.scale_dims)
            self.anchor_last_scale = bool(anchor_last_scale) and self.num_scales > 1
            self.output_scales = self.num_scales - 1 if self.anchor_last_scale else self.num_scales
            hidden = int(hidden_dim or max(96, self.source_dim // 3))
            self.norm = nn.LayerNorm(self.source_dim)
            self.fc1 = nn.Linear(self.source_dim, hidden)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden, self.output_scales)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
            self.delta = max(0.0, float(delta))

        def forward(self, embeddings: list[Any], source_embedding) -> tuple[list[Any], Any]:
            """Predict bounded per-scale residual weights around one from the deepest embedding."""
            if len(embeddings) != self.num_scales:
                raise ValueError(
                    f"ResidualSemanticScaleGate expected {self.num_scales} embeddings, got {len(embeddings)}."
                )
            logits = self.fc2(self.act(self.fc1(self.norm(source_embedding))))
            if self.anchor_last_scale:
                zeros = torch.zeros(
                    (logits.shape[0], 1),
                    device=logits.device,
                    dtype=logits.dtype,
                )
                logits = torch.cat([logits, zeros], dim=1)
            residual = torch.tanh(logits) * self.delta
            weights = 1.0 + residual
            gated = [embedding * weights[:, index : index + 1] for index, embedding in enumerate(embeddings)]
            return gated, weights


    class LesioNeXtBaseClassifier(nn.Module):
        """Base implementation for the LesioNeXt ConvNeXt-based single-model family."""

        def __init__(
            self,
            *,
            backbone_name: str = "convnext_tiny",
            pretrained: bool = True,
            in_chans: int = 3,
            num_classes: int = 2,
            proj_dim: int = 256,
            attn_heads: int = 4,
            attn_mlp_ratio: float = 2.0,
            head_hidden_dim: int = 256,
            attn_dropout: float = 0.1,
            head_dropout: float = 0.2,
            use_stage4_attention: bool = True,
            stage4_attention_type: str = "mhsa",
            stage4_coord_attention_reduction: int = 32,
            stage4_gc_reduction: int = 16,
            stage4_lka_kernel_size: int = 7,
            stage4_lka_dilation: int = 3,
            active_stage_indices: list[int] | tuple[int, ...] | None = None,
            pool_activation: str = "gelu",
            stage_proj_dims: list[int] | tuple[int, ...] | None = None,
            stage_fusion_weights: list[float] | tuple[float, ...] | None = None,
            use_projection_grn: bool = False,
            use_projection_eca: bool = False,
            projection_eca_kernel_size: int | None = None,
            use_projection_spectral_gate: bool = False,
            projection_spectral_gate_stages: list[int] | tuple[int, ...] | None = None,
            projection_spectral_gate_freq_size: int = 4,
            projection_spectral_gate_hidden_dim: int | None = None,
            projection_spectral_gate_include_spatial: bool = True,
            use_domain_invariant_norm: bool = False,
            domain_norm_stages: list[int] | tuple[int, ...] | None = None,
            use_mixstyle: bool = False,
            mixstyle_stages: list[int] | tuple[int, ...] | None = None,
            mixstyle_probability: float = 0.5,
            mixstyle_alpha: float = 0.1,
            use_stage_local_mixer: bool = False,
            stage_local_mixer_stages: list[int] | tuple[int, ...] | None = None,
            stage_local_mixer_square_kernel: int = 3,
            stage_local_mixer_band_kernel_sizes: list[int] | tuple[int, ...] | None = None,
            stage_local_mixer_reduction: int = 16,
            stage_local_mixer_min_channels: int = 32,
            stage_local_mixer_use_identity: bool = True,
            use_scale_gate: bool = False,
            scale_gate_mode: str = "concat",
            scale_gate_hidden_dim: int | None = None,
            scale_gate_temperature: float = 1.0,
            scale_gate_delta: float = 0.15,
            scale_gate_anchor_last: bool = True,
            use_stage_attn_res: bool = False,
            stage_attn_res_mode: str = "content",
            stage_attn_res_attention_dim: int = 128,
            stage_attn_res_temperature: float = 1.0,
            stage_attn_res_strength: float = 1.0,
            stage_attn_res_descriptor_dim: int = 0,
            stage_attn_res_dropout: float = 0.0,
            use_delta_history: bool = False,
            delta_history_mode: str = "inject",
            delta_history_dim: int = 128,
            delta_history_temperature: float = 1.0,
            delta_history_strength: float = 0.5,
            delta_history_descriptor_dim: int = 0,
            learnable_stage_fusion: bool = False,
            fusion_weight_epsilon: float = 1e-4,
            classifier_head_type: str = "linear",
            classifier_head_scale: float = 16.0,
            classifier_head_margin: float = 0.15,
            classifier_head_easy_margin: bool = False,
            **backbone_kwargs,
        ) -> None:
            """Build the LesioNeXt single-model family from a timm ConvNeXt backbone."""
            super().__init__()
            if timm is None:
                raise RuntimeError("timm is required to construct LesioNeXt.")
            self.backbone_name = str(backbone_name)
            self.use_stage4_attention = bool(use_stage4_attention)
            self.use_projection_grn = bool(use_projection_grn)
            self.use_projection_eca = bool(use_projection_eca)
            self.use_projection_spectral_gate = bool(use_projection_spectral_gate)
            self.use_domain_invariant_norm = bool(use_domain_invariant_norm)
            self.use_mixstyle = bool(use_mixstyle)
            self.use_stage_local_mixer = bool(use_stage_local_mixer)
            self.use_scale_gate = bool(use_scale_gate)
            self.use_stage_attn_res = bool(use_stage_attn_res)
            self.use_delta_history = bool(use_delta_history)
            self.delta_history_mode = str(delta_history_mode).lower()
            if self.delta_history_mode not in {"inject", "monitor"}:
                raise ValueError("delta_history_mode must be inject or monitor.")
            if (
                self.use_stage_attn_res
                and self.use_delta_history
                and self.delta_history_mode != "monitor"
            ):
                raise ValueError(
                    "Concurrent stage attention and delta history require delta_history_mode=monitor."
                )
            self.learnable_stage_fusion = bool(learnable_stage_fusion)
            resolved_classifier_head_type = str(classifier_head_type).lower()
            if resolved_classifier_head_type not in {"linear", "arc_margin"}:
                resolved_classifier_head_type = "linear"
            self.classifier_head_type = resolved_classifier_head_type
            resolved_stage4_attention_type = str(stage4_attention_type).lower()
            if resolved_stage4_attention_type not in {"mhsa", "coordatt", "gc", "lka"}:
                resolved_stage4_attention_type = "mhsa"
            self.stage4_attention_type = resolved_stage4_attention_type
            resolved_scale_gate_mode = str(scale_gate_mode).lower()
            if resolved_scale_gate_mode not in {"concat", "stage4", "stage4_residual"}:
                resolved_scale_gate_mode = "concat"
            self.scale_gate_mode = resolved_scale_gate_mode
            self.backbone = timm.create_model(
                self.backbone_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(1, 2, 3),
                in_chans=in_chans,
                **backbone_kwargs,
            )
            channels = list(self.backbone.feature_info.channels())
            if len(channels) != 3:
                raise RuntimeError(
                    f"LesioNeXt expects three feature stages, got {len(channels)} from {self.backbone_name!r}."
                )
            self._stage_numbers = (2, 3, 4)
            self._stage_number_to_position = {
                stage_number: position
                for position, stage_number in enumerate(self._stage_numbers)
            }
            resolved_domain_norm_stages = tuple(
                int(stage_number)
                for stage_number in (domain_norm_stages if domain_norm_stages is not None else (2, 3, 4))
            )
            if any(stage_number not in self._stage_numbers for stage_number in resolved_domain_norm_stages):
                raise ValueError("domain_norm_stages must be selected from stages [2, 3, 4].")
            self.domain_norm_stages = resolved_domain_norm_stages if self.use_domain_invariant_norm else ()
            self.domain_norm_layers = nn.ModuleDict(
                {
                    str(stage_number): nn.GroupNorm(
                        num_groups=1,
                        num_channels=int(channels[self._stage_number_to_position[stage_number]]),
                        affine=True,
                    )
                    for stage_number in self.domain_norm_stages
                }
            )
            self.active_stage_indices = self._normalize_active_stage_indices(active_stage_indices)
            self.stage_proj_dims = self._normalize_stage_proj_dims(
                stage_proj_dims,
                default_dim=int(proj_dim),
            )
            self.stage_fusion_weights = self._normalize_stage_fusion_weights(stage_fusion_weights)
            self.fusion_weight_epsilon = max(float(fusion_weight_epsilon), 1e-8)
            self.stage_local_mixer_stages = self._normalize_stage_local_mixer_stages(
                stage_local_mixer_stages,
            )
            self.projection_spectral_gate_stages = self._normalize_projection_spectral_gate_stages(
                projection_spectral_gate_stages,
            )
            self.mixstyle_stages = self._normalize_mixstyle_stages(mixstyle_stages)
            stage4_dim = int(channels[-1])
            if self.use_stage4_attention and self.stage4_attention_type == "coordatt":
                self.context_block = CoordinateAttention2d(
                    stage4_dim,
                    reduction=int(stage4_coord_attention_reduction),
                )
            elif self.use_stage4_attention and self.stage4_attention_type == "gc":
                self.context_block = GlobalContextBlock2d(
                    stage4_dim,
                    reduction=int(stage4_gc_reduction),
                )
            elif self.use_stage4_attention and self.stage4_attention_type == "lka":
                self.context_block = LargeKernelAttention2d(
                    stage4_dim,
                    kernel_size=int(stage4_lka_kernel_size),
                    dilation=int(stage4_lka_dilation),
                )
            elif self.use_stage4_attention:
                self.context_block = LightweightSelfAttention(
                    stage4_dim,
                    heads=int(attn_heads),
                    dropout=float(attn_dropout),
                    mlp_ratio=float(attn_mlp_ratio),
                )
            else:
                self.context_block = None
            activation_name = str(pool_activation).lower()

            def _projection_activation():
                if activation_name == "gelu":
                    return nn.GELU()
                if activation_name == "softplus":
                    return nn.Softplus()
                return nn.ReLU(inplace=True)

            if activation_name not in {"gelu", "softplus", "relu"}:
                activation_name = "gelu"
            self.pool_activation = activation_name
            self.stage_projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(int(channel), self.stage_proj_dims[position], kernel_size=1, bias=False),
                        _projection_activation(),
                        GlobalResponseNorm2d(self.stage_proj_dims[position]) if self.use_projection_grn else nn.Identity(),
                        EfficientChannelAttention2d(
                            self.stage_proj_dims[position],
                            kernel_size=projection_eca_kernel_size,
                        )
                        if self.use_projection_eca
                        else nn.Identity(),
                    )
                    for position, channel in enumerate(channels)
                ]
            )
            self.projection_spectral_gates = nn.ModuleDict(
                {
                    str(stage_number): SpectralChannelGate2d(
                        self.stage_proj_dims[self._stage_number_to_position[stage_number]],
                        freq_size=int(projection_spectral_gate_freq_size),
                        hidden_dim=projection_spectral_gate_hidden_dim,
                        include_spatial=bool(projection_spectral_gate_include_spatial),
                    )
                    for stage_number in self.projection_spectral_gate_stages
                }
            )
            self.mixstyle_layers = nn.ModuleDict(
                {
                    str(stage_number): MixStyle2d(
                        p=float(mixstyle_probability),
                        alpha=float(mixstyle_alpha),
                    )
                    for stage_number in self.mixstyle_stages
                }
            )
            self.stage_local_mixers = nn.ModuleDict(
                {
                    str(stage_number): SelectiveReceptiveFieldMixer2d(
                        int(channels[self._stage_number_to_position[stage_number]]),
                        square_kernel=int(stage_local_mixer_square_kernel),
                        band_kernel_sizes=stage_local_mixer_band_kernel_sizes,
                        reduction=int(stage_local_mixer_reduction),
                        min_channels=int(stage_local_mixer_min_channels),
                        use_identity=bool(stage_local_mixer_use_identity),
                    )
                    for stage_number in self.stage_local_mixer_stages
                }
            )
            self.stage_pools = nn.ModuleList([GeMPool2d(p=3.0, learnable=True) for _ in channels])
            if self.learnable_stage_fusion:
                self.stage_fusion_logits = nn.Parameter(
                    torch.tensor(self.stage_fusion_weights, dtype=torch.float32)
                )
            else:
                self.register_parameter("stage_fusion_logits", None)
            active_scale_count = len(self.active_stage_indices)
            active_scale_dims = [
                self.stage_proj_dims[self._stage_number_to_position[stage_number]]
                for stage_number in self.active_stage_indices
            ]
            self.scale_gate = (
                (
                    SemanticScaleGate(
                        source_dim=self.stage_proj_dims[self._stage_number_to_position[4]],
                        scale_dims=active_scale_dims,
                        hidden_dim=scale_gate_hidden_dim,
                        temperature=scale_gate_temperature,
                    )
                    if self.scale_gate_mode == "stage4"
                    else ResidualSemanticScaleGate(
                        source_dim=self.stage_proj_dims[self._stage_number_to_position[4]],
                        scale_dims=active_scale_dims,
                        hidden_dim=scale_gate_hidden_dim,
                        delta=scale_gate_delta,
                        anchor_last_scale=scale_gate_anchor_last,
                    )
                    if self.scale_gate_mode == "stage4_residual"
                    else SelectiveScaleGate(
                        active_scale_dims,
                        hidden_dim=scale_gate_hidden_dim,
                        temperature=scale_gate_temperature,
                    )
                )
                if self.use_scale_gate and active_scale_count > 1
                else None
            )
            active_initial_weights = [
                self.stage_fusion_weights[self._stage_number_to_position[stage_number]]
                for stage_number in self.active_stage_indices
            ]
            with torch.random.fork_rng(devices=[]):
                self.stage_attn_res = (
                    StageAttentionResidualFusion(
                        active_scale_dims,
                        attention_dim=int(stage_attn_res_attention_dim),
                        temperature=float(stage_attn_res_temperature),
                        residual_strength=float(stage_attn_res_strength),
                        query_mode=str(stage_attn_res_mode),
                        descriptor_dim=int(stage_attn_res_descriptor_dim),
                        initial_weights=active_initial_weights,
                        dropout=float(stage_attn_res_dropout),
                    )
                    if self.use_stage_attn_res and active_scale_count > 1
                    else None
                )
            with torch.random.fork_rng(devices=[]):
                self.delta_history = (
                    CrossStageDeltaHistoryFusion(
                        active_scale_dims,
                        history_dim=int(delta_history_dim),
                        temperature=float(delta_history_temperature),
                        residual_strength=float(delta_history_strength),
                        descriptor_dim=int(delta_history_descriptor_dim),
                    )
                    if self.use_delta_history and active_scale_count > 1
                    else None
                )
            self.last_scale_gate_weights = None
            self.last_stage_attn_res_weights = None
            self.last_delta_history_weights = None
            self.last_stage_fusion_weights = None
            self.last_projection_spectral_gate_weights = None
            fused_dim = int(sum(active_scale_dims))
            self.head_norm = nn.LayerNorm(fused_dim)
            self.head_dropout = nn.Dropout(float(head_dropout))
            self.head_fc1 = nn.Linear(fused_dim, int(head_hidden_dim))
            self.head_act = nn.GELU()
            self.head_hidden_dropout = nn.Dropout(0.1)
            if self.classifier_head_type == "arc_margin":
                self.classifier = ArcMarginProduct(
                    int(head_hidden_dim),
                    num_classes,
                    scale=float(classifier_head_scale),
                    margin=float(classifier_head_margin),
                    easy_margin=bool(classifier_head_easy_margin),
                )
            else:
                self.classifier = nn.Linear(int(head_hidden_dim), num_classes)
            self.gradcam_layer = self._resolve_gradcam_layer()
            self.feature_info = getattr(self.backbone, "feature_info", None)
            self.pretrained_cfg = dict(getattr(self.backbone, "pretrained_cfg", {}) or {})
            self.default_cfg = dict(getattr(self.backbone, "default_cfg", {}) or {})

        @staticmethod
        def _normalize_active_stage_indices(
            active_stage_indices: list[int] | tuple[int, ...] | None,
        ) -> tuple[int, ...]:
            """Validate and normalize active stage numbers used by the fusion head."""
            if active_stage_indices is None:
                normalized = (2, 3, 4)
            else:
                normalized = tuple(int(index) for index in active_stage_indices)
            if normalized not in VALID_LESIONEXT_STAGE_INDEX_GROUPS:
                raise ValueError(
                    "active_stage_indices must be one of [2, 3, 4], [3, 4], or [4]."
                )
            return normalized

        @staticmethod
        def _normalize_stage_proj_dims(
            stage_proj_dims: list[int] | tuple[int, ...] | None,
            *,
            default_dim: int,
        ) -> tuple[int, int, int]:
            """Return validated per-stage projection dimensions for stages 2, 3, and 4."""
            if stage_proj_dims is None:
                normalized = (int(default_dim), int(default_dim), int(default_dim))
            else:
                normalized = tuple(int(dim) for dim in stage_proj_dims)
            if len(normalized) != 3 or any(dim <= 0 for dim in normalized):
                raise ValueError("stage_proj_dims must contain three positive integers for stages [2, 3, 4].")
            return normalized

        @staticmethod
        def _normalize_stage_fusion_weights(
            stage_fusion_weights: list[float] | tuple[float, ...] | None,
        ) -> tuple[float, float, float]:
            """Return validated per-stage fusion weights for stages 2, 3, and 4."""
            if stage_fusion_weights is None:
                normalized = (1.0, 1.0, 1.0)
            else:
                normalized = tuple(float(weight) for weight in stage_fusion_weights)
            if len(normalized) != 3 or any(weight < 0.0 for weight in normalized):
                raise ValueError("stage_fusion_weights must contain three non-negative scalars for stages [2, 3, 4].")
            return normalized

        @staticmethod
        def _normalize_stage_local_mixer_stages(
            stage_local_mixer_stages: list[int] | tuple[int, ...] | None,
        ) -> tuple[int, ...]:
            """Return validated stage ids that receive selective local mixing."""
            if stage_local_mixer_stages is None:
                normalized = (2, 3)
            else:
                normalized = tuple(int(stage_number) for stage_number in stage_local_mixer_stages)
            if len(normalized) == 0:
                return ()
            if len(set(normalized)) != len(normalized):
                raise ValueError("stage_local_mixer_stages must not contain duplicate stage numbers.")
            valid = {2, 3, 4}
            if any(stage_number not in valid for stage_number in normalized):
                raise ValueError("stage_local_mixer_stages must be selected from stages [2, 3, 4].")
            return normalized

        @staticmethod
        def _normalize_projection_spectral_gate_stages(
            projection_spectral_gate_stages: list[int] | tuple[int, ...] | None,
        ) -> tuple[int, ...]:
            """Return validated projected stage ids that receive spectral gating."""
            if projection_spectral_gate_stages is None:
                normalized = (2, 3, 4)
            else:
                normalized = tuple(int(stage_number) for stage_number in projection_spectral_gate_stages)
            if len(normalized) == 0:
                return ()
            if len(set(normalized)) != len(normalized):
                raise ValueError("projection_spectral_gate_stages must not contain duplicate stage numbers.")
            valid = {2, 3, 4}
            if any(stage_number not in valid for stage_number in normalized):
                raise ValueError("projection_spectral_gate_stages must be selected from stages [2, 3, 4].")
            return normalized

        @staticmethod
        def _normalize_mixstyle_stages(
            mixstyle_stages: list[int] | tuple[int, ...] | None,
        ) -> tuple[int, ...]:
            """Return validated stage ids that receive MixStyle augmentation."""
            if mixstyle_stages is None:
                normalized = (2, 3)
            else:
                normalized = tuple(int(stage_number) for stage_number in mixstyle_stages)
            if len(normalized) == 0:
                return ()
            if len(set(normalized)) != len(normalized):
                raise ValueError("mixstyle_stages must not contain duplicate stage numbers.")
            valid = {2, 3, 4}
            if any(stage_number not in valid for stage_number in normalized):
                raise ValueError("mixstyle_stages must be selected from stages [2, 3, 4].")
            return normalized

        def _resolve_gradcam_layer(self):
            """Select the final projection layer used for Grad-CAM."""
            target_stage = 4 if 4 in self.active_stage_indices else self.active_stage_indices[-1]
            projection_position = self._stage_number_to_position[target_stage]
            return self.stage_projections[projection_position][0]

        def get_gradcam_target_layer(self):
            """Return the preferred Grad-CAM target layer for this wrapper model."""
            return self.gradcam_layer

        def _active_stage_fusion_factors(self, *, device, dtype):
            """Return one multiplicative fusion factor for each active stage."""
            if self.learnable_stage_fusion and self.stage_fusion_logits is not None:
                gathered = torch.stack(
                    [
                        self.stage_fusion_logits[self._stage_number_to_position[stage_number]]
                        for stage_number in self.active_stage_indices
                    ]
                ).to(device=device, dtype=dtype)
                positive = F.relu(gathered) + self.fusion_weight_epsilon
                normalized = positive / positive.sum().clamp(min=self.fusion_weight_epsilon)
                weights = normalized * float(len(self.active_stage_indices))
            else:
                weights = torch.tensor(
                    [
                        self.stage_fusion_weights[self._stage_number_to_position[stage_number]]
                        for stage_number in self.active_stage_indices
                    ],
                    device=device,
                    dtype=dtype,
                )
            self.last_stage_fusion_weights = weights.detach()
            return weights

        def _contextualize_stage4(self, x):
            """Apply one lightweight self-attention block to the last feature stage."""
            if self.context_block is None:
                return x
            if self.stage4_attention_type in {"coordatt", "gc", "lka"}:
                return self.context_block(x)
            batch_size, channels, height, width = x.shape
            tokens = x.flatten(2).transpose(1, 2)
            tokens = self.context_block(tokens)
            return tokens.transpose(1, 2).reshape(batch_size, channels, height, width)

        def _stage_embedding_bundle(self, x, context_descriptor=None):
            """Return active projected stage embeddings before the final classifier head."""
            stage_features = list(self.backbone(x))
            if len(stage_features) != 3:
                raise RuntimeError(f"LesioNeXt received {len(stage_features)} features, expected 3.")
            if self.domain_norm_stages:
                for stage_number in self.domain_norm_stages:
                    position = self._stage_number_to_position[stage_number]
                    stage_features[position] = self.domain_norm_layers[str(stage_number)](stage_features[position])
            if self.use_stage4_attention and 4 in self.active_stage_indices and self.context_block is not None:
                stage4_position = self._stage_number_to_position[4]
                stage_features[stage4_position] = self._contextualize_stage4(stage_features[stage4_position])
            if self.use_mixstyle and len(self.mixstyle_layers) > 0:
                for stage_number in self.mixstyle_stages:
                    position = self._stage_number_to_position[stage_number]
                    stage_features[position] = self.mixstyle_layers[str(stage_number)](stage_features[position])
            if self.use_stage_local_mixer and len(self.stage_local_mixers) > 0:
                for stage_number in self.stage_local_mixer_stages:
                    position = self._stage_number_to_position[stage_number]
                    stage_features[position] = self.stage_local_mixers[str(stage_number)](stage_features[position])
            pooled_embeddings = []
            stage4_embedding = None
            projection_spectral_gate_weights: dict[str, Any] = {}
            fusion_factors = self._active_stage_fusion_factors(
                device=stage_features[0].device,
                dtype=stage_features[0].dtype,
            )
            for active_position, stage_number in enumerate(self.active_stage_indices):
                position = self._stage_number_to_position[stage_number]
                feature = stage_features[position]
                projection = self.stage_projections[position]
                pool = self.stage_pools[position]
                projected = projection(feature)
                if self.use_projection_spectral_gate and str(stage_number) in self.projection_spectral_gates:
                    projected, gate_weights = self.projection_spectral_gates[str(stage_number)](projected)
                    projection_spectral_gate_weights[str(stage_number)] = gate_weights.detach()
                pooled = pool(projected).flatten(1)
                if stage_number == 4:
                    stage4_embedding = pooled
                if self.stage_attn_res is None and self.delta_history is None:
                    pooled = pooled * fusion_factors[active_position]
                pooled_embeddings.append(pooled)
            base_embeddings = pooled_embeddings
            stage_attn_weights = None
            history_weights = None
            if self.stage_attn_res is not None:
                pooled_embeddings, stage_attn_weights = self.stage_attn_res(
                    pooled_embeddings,
                    descriptor=context_descriptor,
                )
            if self.delta_history is not None:
                delta_embeddings, history_weights = self.delta_history(
                    base_embeddings if self.delta_history_mode == "monitor" else pooled_embeddings,
                    descriptor=context_descriptor,
                )
                if self.delta_history_mode == "inject":
                    pooled_embeddings = delta_embeddings
            self.last_stage_attn_res_weights = (
                stage_attn_weights.detach() if stage_attn_weights is not None else None
            )
            self.last_delta_history_weights = (
                history_weights.detach() if history_weights is not None else None
            )
            if stage_attn_weights is not None:
                self.last_stage_fusion_weights = stage_attn_weights.detach()
            elif history_weights is not None:
                self.last_stage_fusion_weights = history_weights.detach()
            if self.scale_gate is not None:
                if self.scale_gate_mode in {"stage4", "stage4_residual"}:
                    if stage4_embedding is None:
                        stage4_embedding = pooled_embeddings[-1]
                    pooled_embeddings, scale_weights = self.scale_gate(pooled_embeddings, stage4_embedding)
                else:
                    pooled_embeddings, scale_weights = self.scale_gate(pooled_embeddings)
                self.last_scale_gate_weights = scale_weights.detach()
            else:
                self.last_scale_gate_weights = None
            self.last_projection_spectral_gate_weights = projection_spectral_gate_weights or None
            stage_embeddings = {
                stage_number: pooled_embeddings[position]
                for position, stage_number in enumerate(self.active_stage_indices)
            }
            return pooled_embeddings, stage_embeddings

        def _fused_embedding(self, x, context_descriptor=None):
            """Return the fused representation before classifier-only dropout."""
            pooled_embeddings, _stage_embeddings = self._stage_embedding_bundle(
                x,
                context_descriptor=context_descriptor,
            )
            fused = torch.cat(pooled_embeddings, dim=1)
            fused = self.head_norm(fused)
            fused = self.head_dropout(fused)
            fused = self.head_fc1(fused)
            fused = self.head_act(fused)
            return fused

        def forward_with_embedding(self, x, labels=None, context_descriptor=None):
            """Return class logits together with the fused training embedding."""
            fused = self._fused_embedding(x, context_descriptor=context_descriptor)
            classifier_input = self.head_hidden_dropout(fused)
            if self.classifier_head_type == "arc_margin":
                logits = self.classifier(classifier_input, labels=labels)
            else:
                logits = self.classifier(classifier_input)
            return logits, fused

        def forward(self, x, labels=None, context_descriptor=None):
            """Return class logits for one image batch."""
            logits, _embedding = self.forward_with_embedding(
                x,
                labels=labels,
                context_descriptor=context_descriptor,
            )
            return logits


    class LesioNeXtClassifier(LesioNeXtBaseClassifier):
        """Preferred public alias for the LesioNeXt single-model family."""


    class DARAClassifier(LesioNeXtClassifier):
        """Historical DARA implementation retained for checkpoint compatibility."""

        def __init__(
            self,
            *,
            num_classes: int = 2,
            use_shared_routed_experts: bool = True,
            shared_expert_hidden_dim: int = 256,
            routed_expert_hidden_dim: int = 128,
            routed_expert_count: int = 4,
            routed_expert_top_k: int = 2,
            routed_expert_router_hidden_dim: int = 128,
            routed_expert_temperature: float = 1.0,
            routed_expert_scale: float = 0.5,
            routed_expert_dropout: float = 0.1,
            use_reliability_gate: bool = False,
            reliability_threshold: float = 0.15,
            reliability_temperature: float = 0.1,
            **model_kwargs,
        ) -> None:
            """Build the DARA base and configurable shared-routed expert head."""
            model_kwargs = dict(model_kwargs)
            model_kwargs.setdefault("use_stage_attn_res", True)
            if use_reliability_gate:
                model_kwargs.setdefault("use_delta_history", True)
                model_kwargs.setdefault("delta_history_mode", "monitor")
            model_kwargs.setdefault("classifier_head_type", "linear")
            super().__init__(num_classes=num_classes, **model_kwargs)
            self.use_shared_routed_experts = bool(use_shared_routed_experts)
            self.use_reliability_gate = bool(use_reliability_gate)
            self.last_expert_weights = None
            self.last_dense_expert_weights = None
            self.last_shared_logits = None
            self.last_routed_logits = None
            self.last_routing_confidence = None
            self.last_history_confidence = None
            self.last_combined_reliability = None
            self.last_reliability_gate = None
            if self.use_shared_routed_experts:
                head_class = (
                    ReliabilityGatedSharedRoutedExpertHead
                    if self.use_reliability_gate
                    else SharedRoutedExpertHead
                )
                head_kwargs = {}
                if self.use_reliability_gate:
                    head_kwargs = {
                        "reliability_threshold": float(reliability_threshold),
                        "reliability_temperature": float(reliability_temperature),
                        "use_reliability_gate": True,
                    }
                self.classifier = head_class(
                    self.head_fc1.out_features,
                    num_classes,
                    shared_hidden_dim=int(shared_expert_hidden_dim),
                    expert_hidden_dim=int(routed_expert_hidden_dim),
                    router_hidden_dim=int(routed_expert_router_hidden_dim),
                    num_routed_experts=int(routed_expert_count),
                    top_k=int(routed_expert_top_k),
                    router_temperature=float(routed_expert_temperature),
                    routed_scale=float(routed_expert_scale),
                    dropout=float(routed_expert_dropout),
                    **head_kwargs,
                )

        def _history_reliability(self, reference):
            """Convert deepest-stage delta-history concentration into reliability."""
            weights = self.last_delta_history_weights
            if weights is None:
                return reference.new_zeros((reference.shape[0],))
            deepest = weights[:, -1, :].to(device=reference.device, dtype=reference.dtype)
            entropy = -(deepest.clamp_min(1e-8) * deepest.clamp_min(1e-8).log()).sum(dim=1)
            max_entropy = math.log(float(deepest.shape[1]))
            return (1.0 - entropy / max(max_entropy, 1e-8)).clamp(0.0, 1.0)

        def forward_with_embedding(self, x, labels=None, context_descriptor=None):
            """Return DARA logits, embedding, and routing diagnostics."""
            fused = self._fused_embedding(x, context_descriptor=context_descriptor)
            classifier_input = self.head_hidden_dropout(fused)
            if self.use_shared_routed_experts and self.use_reliability_gate:
                history_reliability = self._history_reliability(classifier_input)
                logits = self.classifier(
                    classifier_input,
                    external_reliability=history_reliability,
                )
                self.last_history_confidence = history_reliability.detach()
            else:
                logits = self.classifier(classifier_input)
                self.last_history_confidence = None
            if self.use_shared_routed_experts:
                self.last_dense_expert_weights = getattr(
                    self.classifier,
                    "last_dense_expert_weights",
                    None,
                )
                self.last_expert_weights = self.classifier.last_expert_weights
                self.last_shared_logits = self.classifier.last_shared_logits
                self.last_routed_logits = self.classifier.last_routed_logits
                self.last_routing_confidence = getattr(
                    self.classifier,
                    "last_routing_confidence",
                    None,
                )
                self.last_combined_reliability = getattr(
                    self.classifier,
                    "last_combined_reliability",
                    None,
                )
                self.last_reliability_gate = getattr(
                    self.classifier,
                    "last_reliability_gate",
                    None,
                )
            return logits, fused


    class AERISClassifier(DARAClassifier):
        """Official AERIS classifier with inter-stage selection and expert routing."""


    class QDHEClassifier(DARAClassifier):
        """Backward-compatible alias for the DARA delta-history gated ablation."""

        def __init__(
            self,
            *,
            num_classes: int = 2,
            **model_kwargs,
        ) -> None:
            """Map the legacy name to the full DARA enhancement switches."""
            model_kwargs = dict(model_kwargs)
            model_kwargs.setdefault("use_stage_attn_res", True)
            model_kwargs.setdefault("use_delta_history", True)
            model_kwargs.setdefault("delta_history_mode", "monitor")
            model_kwargs.setdefault("use_shared_routed_experts", True)
            model_kwargs.setdefault("use_reliability_gate", True)
            super().__init__(
                num_classes=num_classes,
                **model_kwargs,
            )


    class LesionScaleMoEConvNeXtClassifier(LesioNeXtClassifier):
        """LesioNeXt-MoE classifier with soft lesion-scale expert routing."""

        input_mode = "dual_view_roi"

        def __init__(
            self,
            *,
            num_classes: int = 2,
            descriptor_dim: int = 14,
            expert_hidden_dim: int = 256,
            router_hidden_dim: int = 160,
            router_temperature: float = 1.0,
            fixed_equal_expert_weights: bool = False,
            router_uses_descriptors: bool = True,
            expert_dropout: float | None = None,
            **model_kwargs,
        ) -> None:
            """Build a three-expert lesion-scale MoE on top of LesioNeXt stage embeddings."""
            model_kwargs = dict(model_kwargs)
            model_kwargs.setdefault("active_stage_indices", [2, 3, 4])
            model_kwargs.setdefault("stage_proj_dims", [192, 256, 320])
            model_kwargs.setdefault("stage_fusion_weights", [0.8, 1.0, 1.0])
            model_kwargs.setdefault("use_stage4_attention", False)
            model_kwargs.setdefault("use_projection_grn", True)
            model_kwargs.setdefault("use_projection_eca", True)
            super().__init__(num_classes=num_classes, **model_kwargs)
            missing_stages = {2, 3, 4}.difference(self.active_stage_indices)
            if missing_stages:
                raise ValueError("LesionScaleMoE requires active_stage_indices to include [2, 3, 4].")
            self.descriptor_dim = int(descriptor_dim)
            self.num_experts = 3
            self.fixed_equal_expert_weights = bool(fixed_equal_expert_weights)
            self.router_uses_descriptors = bool(router_uses_descriptors)
            self.router_temperature = max(float(router_temperature), 1e-4)
            self.last_expert_weights = None
            stage2_dim = self.stage_proj_dims[self._stage_number_to_position[2]]
            stage3_dim = self.stage_proj_dims[self._stage_number_to_position[3]]
            stage4_dim = self.stage_proj_dims[self._stage_number_to_position[4]]
            self._moe_stage_dims = {2: stage2_dim, 3: stage3_dim, 4: stage4_dim}
            fused_dim = int(stage2_dim + stage3_dim + stage4_dim)
            dropout = float(self.head_dropout.p if expert_dropout is None else expert_dropout)
            self.small_expert = self._make_expert_head(stage2_dim + stage3_dim, expert_hidden_dim, num_classes, dropout)
            self.medium_expert = self._make_expert_head(fused_dim, expert_hidden_dim, num_classes, dropout)
            self.large_expert = self._make_expert_head(stage4_dim, expert_hidden_dim, num_classes, dropout)
            descriptor_router_dim = self.descriptor_dim if self.router_uses_descriptors else 0
            router_input_dim = fused_dim * 2 + descriptor_router_dim
            self.router = nn.Sequential(
                nn.LayerNorm(router_input_dim),
                nn.Linear(router_input_dim, int(router_hidden_dim)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(int(router_hidden_dim), self.num_experts),
            )

        @staticmethod
        def _make_expert_head(input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
            """Create one lightweight MLP expert classifier head."""
            return nn.Sequential(
                nn.LayerNorm(int(input_dim)),
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), int(num_classes)),
            )

        def _normalize_descriptors(self, descriptor, reference):
            """Return ROI descriptors aligned to the current image batch."""
            if not self.router_uses_descriptors:
                return reference.new_zeros((reference.shape[0], 0))
            if descriptor is None:
                return reference.new_zeros((reference.shape[0], self.descriptor_dim))
            descriptor = descriptor.to(device=reference.device, dtype=reference.dtype)
            if descriptor.ndim == 1:
                descriptor = descriptor.unsqueeze(0)
            if descriptor.shape[0] == 1 and reference.shape[0] > 1:
                descriptor = descriptor.expand(reference.shape[0], -1)
            if descriptor.shape[1] != self.descriptor_dim:
                raise ValueError(
                    f"LesionScaleMoE expected {self.descriptor_dim} ROI descriptors, got {descriptor.shape[1]}."
                )
            return descriptor

        def _stage_cat(self, stage_embeddings: dict[int, Any]):
            """Concatenate stage 2, 3, and 4 embeddings in anatomical scale order."""
            return torch.cat([stage_embeddings[2], stage_embeddings[3], stage_embeddings[4]], dim=1)

        def _average_stage_embeddings(self, first: dict[int, Any], second: dict[int, Any]) -> dict[int, Any]:
            """Average matching stage embeddings from full-image and ROI views."""
            return {
                stage_number: (first[stage_number] + second[stage_number]) * 0.5
                for stage_number in (2, 3, 4)
            }

        def _resolve_inputs(self, image=None, image_full=None, image_roi=None, roi_descriptor=None, **batch):
            """Normalize supported single-image and dual-view calling conventions."""
            if isinstance(image, dict):
                batch = {**image, **batch}
                image = None
            if image_full is None:
                image_full = batch.get("image_full")
            if image_roi is None:
                image_roi = batch.get("image_roi")
            if roi_descriptor is None:
                roi_descriptor = batch.get("roi_descriptor")
            if image is None:
                image = batch.get("image")
            if image_full is None:
                image_full = image
            if image_full is None:
                raise ValueError("LesionScaleMoE requires image_full or image.")
            return image_full, image_roi, roi_descriptor

        def forward_with_embedding(
            self,
            image=None,
            image_full=None,
            image_roi=None,
            roi_descriptor=None,
            labels=None,
            **batch,
        ):
            """Return logits and router embedding for lesion-scale MoE training."""
            image_full, image_roi, roi_descriptor = self._resolve_inputs(
                image=image,
                image_full=image_full,
                image_roi=image_roi,
                roi_descriptor=roi_descriptor,
                **batch,
            )
            _full_pooled, full_stages = self._stage_embedding_bundle(image_full)
            if image_roi is None:
                roi_stages = full_stages
                full_roi_delta = self._stage_cat(full_stages).new_zeros(self._stage_cat(full_stages).shape)
            else:
                _roi_pooled, roi_stages = self._stage_embedding_bundle(image_roi)
                full_roi_delta = torch.abs(self._stage_cat(full_stages) - self._stage_cat(roi_stages))
            medium_stages = self._average_stage_embeddings(full_stages, roi_stages)
            small_input = torch.cat([roi_stages[2], roi_stages[3]], dim=1)
            medium_input = self._stage_cat(medium_stages)
            large_input = full_stages[4]
            expert_logits = torch.stack(
                [
                    self.small_expert(small_input),
                    self.medium_expert(medium_input),
                    self.large_expert(large_input),
                ],
                dim=1,
            )
            descriptor = self._normalize_descriptors(roi_descriptor, medium_input)
            router_input = torch.cat([self._stage_cat(full_stages), full_roi_delta, descriptor], dim=1)
            if self.fixed_equal_expert_weights:
                expert_weights = router_input.new_full(
                    (router_input.shape[0], self.num_experts),
                    1.0 / float(self.num_experts),
                )
            else:
                expert_weights = torch.softmax(self.router(router_input) / self.router_temperature, dim=1)
            self.last_expert_weights = expert_weights.detach()
            logits = (expert_logits * expert_weights.unsqueeze(-1)).sum(dim=1)
            return logits, router_input

        def forward(
            self,
            image=None,
            image_full=None,
            image_roi=None,
            roi_descriptor=None,
            labels=None,
            **batch,
        ):
            """Return class logits from the lesion-scale MoE classifier."""
            logits, _embedding = self.forward_with_embedding(
                image=image,
                image_full=image_full,
                image_roi=image_roi,
                roi_descriptor=roi_descriptor,
                labels=labels,
                **batch,
            )
            return logits


    class LesionScaleMoEV2ConvNeXtClassifier(LesionScaleMoEConvNeXtClassifier):
        """LesioNeXt-MoE v2 with area prior, spatial small expert, and residual logits."""

        def __init__(
            self,
            *,
            num_classes: int = 2,
            area_prior_strength: float = 0.75,
            area_prior_small_center: float = 0.12,
            area_prior_medium_center: float = 0.32,
            area_prior_large_center: float = 0.68,
            area_prior_sigma: float = 0.22,
            small_spatial_hidden_dim: int = 96,
            residual_alpha: float = 0.15,
            **model_kwargs,
        ) -> None:
            """Build the v2 MoE classifier while preserving the v1 call surface."""
            super().__init__(num_classes=num_classes, **model_kwargs)
            self.area_prior_strength = float(area_prior_strength)
            self.area_prior_sigma = max(float(area_prior_sigma), 1e-4)
            self.residual_alpha = float(residual_alpha)
            self.register_buffer(
                "area_prior_centers",
                torch.tensor(
                    [
                        float(area_prior_small_center),
                        float(area_prior_medium_center),
                        float(area_prior_large_center),
                    ],
                    dtype=torch.float32,
                ),
                persistent=False,
            )
            stage2_dim = self._moe_stage_dims[2]
            stage3_dim = self._moe_stage_dims[3]
            spatial_hidden_dim = int(small_spatial_hidden_dim)
            self.small_spatial_head = nn.Sequential(
                nn.Conv2d(stage2_dim + stage3_dim, spatial_hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.LayerNorm(spatial_hidden_dim),
            )
            self.small_spatial_logits = nn.Linear(spatial_hidden_dim, num_classes)
            fused_dim = sum(self._moe_stage_dims.values())
            self.baseline_residual_head = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Linear(fused_dim, int(num_classes)),
            )
            self.last_moe_logits = None
            self.last_baseline_logits = None
            self.last_router_area_prior_logits = None

        def _projected_stage_feature_bundle(self, x):
            """Return projected feature maps and pooled embeddings for stages 2, 3, and 4."""
            stage_features = list(self.backbone(x))
            if len(stage_features) != 3:
                raise RuntimeError(f"LesionScaleMoEV2 received {len(stage_features)} features, expected 3.")
            if self.use_stage4_attention and 4 in self.active_stage_indices and self.context_block is not None:
                stage4_position = self._stage_number_to_position[4]
                stage_features[stage4_position] = self._contextualize_stage4(stage_features[stage4_position])
            projected_features = {}
            pooled_embeddings = []
            fusion_factors = self._active_stage_fusion_factors(
                device=stage_features[0].device,
                dtype=stage_features[0].dtype,
            )
            for active_position, stage_number in enumerate(self.active_stage_indices):
                position = self._stage_number_to_position[stage_number]
                projected = self.stage_projections[position](stage_features[position])
                projected_features[stage_number] = projected
                pooled = self.stage_pools[position](projected).flatten(1)
                pooled_embeddings.append(pooled * fusion_factors[active_position])
            stage_embeddings = {
                stage_number: pooled_embeddings[position]
                for position, stage_number in enumerate(self.active_stage_indices)
            }
            return projected_features, stage_embeddings

        def _small_spatial_expert_logits(self, roi_projected_features: dict[int, Any]):
            """Classify small-lesion local details from stage2/stage3 feature maps."""
            stage2 = roi_projected_features[2]
            stage3 = roi_projected_features[3]
            if stage3.shape[-2:] != stage2.shape[-2:]:
                stage3 = F.interpolate(stage3, size=stage2.shape[-2:], mode="bilinear", align_corners=False)
            spatial_features = self.small_spatial_head(torch.cat([stage2, stage3], dim=1))
            return self.small_spatial_logits(spatial_features)

        def _router_area_prior_logits(self, descriptor, reference):
            """Return geometry-derived router prior logits for small/medium/large experts."""
            if self.area_prior_strength <= 0.0:
                return reference.new_zeros((reference.shape[0], self.num_experts))
            descriptor = self._normalize_descriptors(descriptor, reference)
            if descriptor.shape[1] <= 1:
                return reference.new_zeros((reference.shape[0], self.num_experts))
            area_ratio = descriptor[:, 1].clamp(0.0, 1.0).unsqueeze(1)
            centers = self.area_prior_centers.to(device=reference.device, dtype=reference.dtype).view(1, -1)
            distances = (area_ratio - centers).pow(2)
            prior_logits = -distances / (2.0 * self.area_prior_sigma * self.area_prior_sigma)
            return prior_logits * float(self.area_prior_strength)

        def forward_with_embedding(
            self,
            image=None,
            image_full=None,
            image_roi=None,
            roi_descriptor=None,
            labels=None,
            **batch,
        ):
            """Return logits and router embedding for the optimized lesion-scale MoE."""
            image_full, image_roi, roi_descriptor = self._resolve_inputs(
                image=image,
                image_full=image_full,
                image_roi=image_roi,
                roi_descriptor=roi_descriptor,
                **batch,
            )
            _full_projected, full_stages = self._projected_stage_feature_bundle(image_full)
            if image_roi is None:
                roi_projected = _full_projected
                roi_stages = full_stages
                full_stage_cat = self._stage_cat(full_stages)
                full_roi_delta = full_stage_cat.new_zeros(full_stage_cat.shape)
            else:
                roi_projected, roi_stages = self._projected_stage_feature_bundle(image_roi)
                full_stage_cat = self._stage_cat(full_stages)
                full_roi_delta = torch.abs(full_stage_cat - self._stage_cat(roi_stages))
            medium_stages = self._average_stage_embeddings(full_stages, roi_stages)
            small_input = torch.cat([roi_stages[2], roi_stages[3]], dim=1)
            medium_input = self._stage_cat(medium_stages)
            large_input = full_stages[4]
            small_logits = self.small_expert(small_input) + self._small_spatial_expert_logits(roi_projected)
            expert_logits = torch.stack(
                [
                    small_logits,
                    self.medium_expert(medium_input),
                    self.large_expert(large_input),
                ],
                dim=1,
            )
            descriptor = self._normalize_descriptors(roi_descriptor, medium_input)
            router_input = torch.cat([full_stage_cat, full_roi_delta, descriptor], dim=1)
            area_prior_logits = self._router_area_prior_logits(roi_descriptor, medium_input)
            if self.fixed_equal_expert_weights:
                expert_weights = router_input.new_full(
                    (router_input.shape[0], self.num_experts),
                    1.0 / float(self.num_experts),
                )
            else:
                router_logits = self.router(router_input) + area_prior_logits
                expert_weights = torch.softmax(router_logits / self.router_temperature, dim=1)
            moe_logits = (expert_logits * expert_weights.unsqueeze(-1)).sum(dim=1)
            baseline_logits = self.baseline_residual_head(self._stage_cat(medium_stages))
            logits = moe_logits + float(self.residual_alpha) * baseline_logits
            self.last_expert_weights = expert_weights.detach()
            self.last_moe_logits = moe_logits.detach()
            self.last_baseline_logits = baseline_logits.detach()
            self.last_router_area_prior_logits = area_prior_logits.detach()
            return logits, router_input


    class LesionScaleMoEV3ConvNeXtClassifier(LesionScaleMoEV2ConvNeXtClassifier):
        """LesioNeXt-MoE v3 with ROI-valid routing safeguards and anti-collapse expert mixing."""

        def __init__(
            self,
            *,
            num_classes: int = 2,
            max_valid_area_ratio: float = 0.75,
            expert_weight_floor: float = 0.05,
            **model_kwargs,
        ) -> None:
            """Build the v3 MoE classifier while keeping the v2 feature path."""
            super().__init__(num_classes=num_classes, **model_kwargs)
            self.max_valid_area_ratio = float(max_valid_area_ratio)
            floor = max(0.0, float(expert_weight_floor))
            self.expert_weight_floor = min(floor, (1.0 / float(self.num_experts)) - 1e-6)
            self.last_router_valid_mask = None

        def _valid_roi_mask(self, descriptor, reference):
            """Return a [B,1] mask for descriptors that should influence ROI routing."""
            descriptor = self._normalize_descriptors(descriptor, reference)
            if descriptor.shape[1] <= 1:
                return reference.new_zeros((reference.shape[0], 1))
            roi_valid = descriptor[:, 0:1] >= 0.5
            area_valid = descriptor[:, 1:2] <= float(self.max_valid_area_ratio)
            return (roi_valid & area_valid).to(dtype=reference.dtype)

        def _router_descriptor(self, descriptor, reference):
            """Zero geometry descriptors when ROI fallback made them unreliable."""
            descriptor = self._normalize_descriptors(descriptor, reference)
            valid_mask = self._valid_roi_mask(descriptor, reference)
            return descriptor * valid_mask

        def _router_area_prior_logits(self, descriptor, reference):
            """Return area-prior logits only for valid ROI descriptors."""
            prior_logits = super()._router_area_prior_logits(descriptor, reference)
            valid_mask = self._valid_roi_mask(descriptor, reference)
            return prior_logits * valid_mask

        def _mix_expert_weights(self, expert_weights):
            """Keep every expert lightly active to avoid near one-hot large-expert collapse."""
            floor = float(self.expert_weight_floor)
            if floor <= 0.0:
                return expert_weights
            scale = max(0.0, 1.0 - floor * float(self.num_experts))
            return expert_weights * scale + floor

        def forward_with_embedding(
            self,
            image=None,
            image_full=None,
            image_roi=None,
            roi_descriptor=None,
            labels=None,
            **batch,
        ):
            """Return logits and router embedding for ROI-robust lesion-scale MoE."""
            image_full, image_roi, roi_descriptor = self._resolve_inputs(
                image=image,
                image_full=image_full,
                image_roi=image_roi,
                roi_descriptor=roi_descriptor,
                **batch,
            )
            full_projected, full_stages = self._projected_stage_feature_bundle(image_full)
            full_stage_cat = self._stage_cat(full_stages)
            if image_roi is None:
                roi_projected = full_projected
                roi_stages = full_stages
                full_roi_delta = full_stage_cat.new_zeros(full_stage_cat.shape)
            else:
                roi_projected, roi_stages = self._projected_stage_feature_bundle(image_roi)
                full_roi_delta = torch.abs(full_stage_cat - self._stage_cat(roi_stages))

            medium_stages = self._average_stage_embeddings(full_stages, roi_stages)
            small_input = torch.cat([roi_stages[2], roi_stages[3]], dim=1)
            medium_input = self._stage_cat(medium_stages)
            large_input = full_stages[4]

            descriptor = self._normalize_descriptors(roi_descriptor, medium_input)
            valid_mask = self._valid_roi_mask(descriptor, medium_input)
            small_spatial_logits = self._small_spatial_expert_logits(roi_projected)
            full_spatial_logits = self._small_spatial_expert_logits(full_projected)
            small_spatial_logits = small_spatial_logits * valid_mask + full_spatial_logits * (1.0 - valid_mask)
            small_logits = self.small_expert(small_input) + small_spatial_logits
            expert_logits = torch.stack(
                [
                    small_logits,
                    self.medium_expert(medium_input),
                    self.large_expert(large_input),
                ],
                dim=1,
            )

            router_descriptor = self._router_descriptor(descriptor, medium_input)
            router_input = torch.cat([full_stage_cat, full_roi_delta * valid_mask, router_descriptor], dim=1)
            area_prior_logits = self._router_area_prior_logits(descriptor, medium_input)
            if self.fixed_equal_expert_weights:
                expert_weights = router_input.new_full(
                    (router_input.shape[0], self.num_experts),
                    1.0 / float(self.num_experts),
                )
            else:
                router_logits = self.router(router_input) + area_prior_logits
                expert_weights = torch.softmax(router_logits / self.router_temperature, dim=1)
                expert_weights = self._mix_expert_weights(expert_weights)
            moe_logits = (expert_logits * expert_weights.unsqueeze(-1)).sum(dim=1)
            baseline_logits = self.baseline_residual_head(self._stage_cat(medium_stages))
            logits = moe_logits + float(self.residual_alpha) * baseline_logits
            self.last_expert_weights = expert_weights.detach()
            self.last_moe_logits = moe_logits.detach()
            self.last_baseline_logits = baseline_logits.detach()
            self.last_router_area_prior_logits = area_prior_logits.detach()
            self.last_router_valid_mask = valid_mask.detach()
            return logits, router_input


    class DualViewConvNeXtClassifier(nn.Module):
        """Shared-backbone ConvNeXt classifier over full-image and ROI views."""

        input_mode = "dual_view_roi"

        def __init__(
            self,
            *,
            backbone_name: str = "convnext_tiny",
            pretrained: bool = True,
            in_chans: int = 3,
            num_classes: int = 2,
            descriptor_dim: int = 14,
            fusion_hidden_dim: int = 512,
            dropout: float = 0.2,
            **backbone_kwargs,
        ) -> None:
            """Build a shared ConvNeXt encoder plus late fusion classifier head."""
            super().__init__()
            if timm is None:
                raise RuntimeError("timm is required to construct DualViewConvNeXtClassifier.")
            self.backbone_name = str(backbone_name)
            self.descriptor_dim = int(descriptor_dim)
            self.input_mode = "dual_view_roi"
            self.backbone = timm.create_model(
                self.backbone_name,
                pretrained=pretrained,
                in_chans=in_chans,
                num_classes=0,
                global_pool="avg",
                **backbone_kwargs,
            )
            self.pretrained_cfg = getattr(self.backbone, "pretrained_cfg", {}) or {}
            embedding_dim = int(getattr(self.backbone, "num_features", 0))
            if embedding_dim <= 0:
                raise RuntimeError(f"Backbone {self.backbone_name!r} did not expose num_features.")
            fusion_dim = embedding_dim * 4 + self.descriptor_dim
            hidden_dim = int(fusion_hidden_dim)
            self.fusion = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, num_classes),
            )

        def encode(self, image):
            """Return one embedding per image."""
            return self.backbone(image)

        def _normalize_descriptors(self, descriptor, reference):
            """Return descriptor tensor aligned to the image batch."""
            if descriptor is None:
                return reference.new_zeros((reference.shape[0], self.descriptor_dim))
            descriptor = descriptor.to(device=reference.device, dtype=reference.dtype)
            if descriptor.ndim == 1:
                descriptor = descriptor.unsqueeze(0)
            if descriptor.shape[0] == 1 and reference.shape[0] > 1:
                descriptor = descriptor.expand(reference.shape[0], -1)
            if descriptor.shape[1] != self.descriptor_dim:
                raise ValueError(
                    f"DualViewConvNeXt expected {self.descriptor_dim} ROI descriptors, got {descriptor.shape[1]}."
                )
            return descriptor

        def forward_with_embedding(
            self,
            image_full=None,
            image_roi=None,
            roi_descriptor=None,
            labels=None,
            **batch,
        ):
            """Return logits and fused embedding for dual-view training."""
            if isinstance(image_full, dict):
                batch = image_full
                image_full = batch.get("image_full")
                image_roi = batch.get("image_roi")
                roi_descriptor = batch.get("roi_descriptor")
            if image_full is None:
                image_full = batch.get("image")
            if image_full is None:
                raise ValueError("DualViewConvNeXt requires image_full or image.")
            if image_roi is None:
                image_roi = image_full
            full_embedding = self.encode(image_full)
            roi_embedding = self.encode(image_roi)
            descriptors = self._normalize_descriptors(roi_descriptor, full_embedding)
            fused = torch.cat(
                [
                    full_embedding,
                    roi_embedding,
                    torch.abs(full_embedding - roi_embedding),
                    full_embedding * roi_embedding,
                    descriptors,
                ],
                dim=1,
            )
            return self.fusion(fused), fused

        def forward(
            self,
            image_full=None,
            image_roi=None,
            roi_descriptor=None,
            labels=None,
            **batch,
        ):
            """Return class logits for one dual-view batch."""
            logits, _embedding = self.forward_with_embedding(
                image_full=image_full,
                image_roi=image_roi,
                roi_descriptor=roi_descriptor,
                labels=labels,
                **batch,
            )
            return logits
else:  # pragma: no cover - torch missing
    class TinyCNNClassifier:  # type: ignore[override]
        """Placeholder classifier used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct the classifier.")


    class GeMPool2d:  # type: ignore[override]
        """Placeholder GeM pooling used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct GeMPool2d.")


    class LightweightSelfAttention:  # type: ignore[override]
        """Placeholder attention block used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct LightweightSelfAttention.")


    class GlobalResponseNorm2d:  # type: ignore[override]
        """Placeholder GRN block used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct GlobalResponseNorm2d.")


    class EfficientChannelAttention2d:  # type: ignore[override]
        """Placeholder ECA block used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct EfficientChannelAttention2d.")


    class SpectralChannelGate2d:  # type: ignore[override]
        """Placeholder spectral gate used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct SpectralChannelGate2d.")


    class CoordinateAttention2d:  # type: ignore[override]
        """Placeholder coordinate attention block used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct CoordinateAttention2d.")


    class SelectiveReceptiveFieldMixer2d:  # type: ignore[override]
        """Placeholder selective receptive-field mixer used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct SelectiveReceptiveFieldMixer2d.")


    class GlobalContextBlock2d:  # type: ignore[override]
        """Placeholder global context block used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct GlobalContextBlock2d.")


    class LargeKernelAttention2d:  # type: ignore[override]
        """Placeholder large-kernel attention block used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct LargeKernelAttention2d.")


    class SelectiveScaleGate:  # type: ignore[override]
        """Placeholder scale gate used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct SelectiveScaleGate.")


    class SemanticScaleGate:  # type: ignore[override]
        """Placeholder semantic scale gate used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct SemanticScaleGate.")


    class ResidualSemanticScaleGate:  # type: ignore[override]
        """Placeholder residual semantic scale gate used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct ResidualSemanticScaleGate.")


    class LesioNeXtBaseClassifier:  # type: ignore[override]
        """Placeholder base implementation for the LesioNeXt single-model family."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct LesioNeXt.")


    class LesioNeXtClassifier(LesioNeXtBaseClassifier):  # type: ignore[override]
        """Preferred placeholder alias for the LesioNeXt single-model family."""


    class DARAClassifier(LesioNeXtClassifier):  # type: ignore[override]
        """Placeholder DARA classifier used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct DARAClassifier.")


    class AERISClassifier(DARAClassifier):  # type: ignore[override]
        """Placeholder AERIS classifier used when torch is unavailable."""


    class QDHEClassifier(LesioNeXtClassifier):  # type: ignore[override]
        """Placeholder QDHE classifier used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct QDHEClassifier.")


    class LesionScaleMoEConvNeXtClassifier:  # type: ignore[override]
        """Placeholder lesion-scale MoE used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct LesionScaleMoEConvNeXtClassifier.")


    class LesionScaleMoEV2ConvNeXtClassifier:  # type: ignore[override]
        """Placeholder lesion-scale MoE v2 used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct LesionScaleMoEV2ConvNeXtClassifier.")


    class LesionScaleMoEV3ConvNeXtClassifier:  # type: ignore[override]
        """Placeholder lesion-scale MoE v3 used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct LesionScaleMoEV3ConvNeXtClassifier.")


    class DualViewConvNeXtClassifier:  # type: ignore[override]
        """Placeholder dual-view ConvNeXt used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct DualViewConvNeXtClassifier.")


def create_classifier(
    model_name: str = "resnet18",
    *,
    pretrained: bool = True,
    in_chans: int = 3,
    num_classes: int = 2,
    **model_kwargs,
):
    """Create a classifier from local aliases, torchvision, timm, or a tiny fallback."""
    require_dependency("torch", torch)
    require_dependency("torch.nn", nn)
    normalized_name = model_name.lower()
    if normalized_name in {"basic_cnn", "tiny_cnn"}:
        return TinyCNNClassifier(in_chans=in_chans, num_classes=num_classes)
    if normalized_name in {
        "lesionext_attnres_tiny",
        "lesionext_v4_4_attnres_tiny",
        "lesionext_block_attnres_tiny",
    }:
        model_kwargs = dict(model_kwargs)
        attnres_defaults = {
            "backbone_name": "convnext_tiny",
            "attnres_stages": [3, 4],
            "attnres_block_size": 2,
            "attnres_inject_every": 1,
            "attnres_temperature": 1.0,
            "attnres_gate_init": 0.0,
            "attnres_query_mode": "pseudo",
            "attnres_fusion_mode": "add",
            "attnres_history_mode": "delta",
            "head_dropout": 0.2,
        }
        attnres_defaults.update(model_kwargs)
        backbone_name = str(attnres_defaults.pop("backbone_name"))
        return DepthwiseBlockAttnResConvNeXtClassifier(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **attnres_defaults,
        )
    if normalized_name in {
        "lesionext_lens_tiny",
        "lesionext_lesion_evidence_tiny",
    }:
        model_kwargs = dict(model_kwargs)
        backbone_name = str(model_kwargs.pop("backbone_name", "convnext_tiny"))
        return LesionEvidenceConvNeXtClassifier(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **model_kwargs,
        )
    full_dara_aliases = {
        "dara_full_tiny",
        "dara_dh_moe_tiny",
        "dara_full_densenet121",
        "dara_dh_moe_densenet121",
    }
    aeris_full_aliases = {
        "aeris",
        "aeris_t",
        "aeris_full",
        "aeris_full_t",
        "aeris_cls",
        "aeris_cls_t",
        "aeris_densenet121",
    }
    legacy_qdhe_aliases = {"qdhe_tiny", "qdhe_moe_tiny", "qdhe", "qdhe_densenet121"}
    if normalized_name in aeris_full_aliases | full_dara_aliases | legacy_qdhe_aliases:
        model_kwargs = dict(model_kwargs)
        is_densenet_variant = normalized_name in {
            "qdhe_densenet121",
            "dara_full_densenet121",
            "dara_dh_moe_densenet121",
            "aeris_densenet121",
        }
        full_dara_defaults = {
            "stage_proj_dims": [96, 128, 160] if is_densenet_variant else [192, 256, 320],
            "stage_fusion_weights": [0.8, 1.0, 1.0],
            "head_hidden_dim": 192 if is_densenet_variant else 320,
            "head_dropout": 0.2,
            "active_stage_indices": [2, 3, 4],
            "use_stage4_attention": False,
            "use_projection_grn": True,
            "use_projection_eca": True,
            "use_stage_local_mixer": not is_densenet_variant,
            "stage_local_mixer_stages": [2, 3],
            "use_scale_gate": False,
            "use_stage_attn_res": True,
            "use_delta_history": True,
            "delta_history_mode": "monitor",
            "delta_history_dim": 128,
            "delta_history_temperature": 0.8,
            "delta_history_strength": 0.5,
            "shared_expert_hidden_dim": 256,
            "routed_expert_hidden_dim": 128,
            "routed_expert_count": 4,
            "routed_expert_top_k": 2,
            "routed_expert_router_hidden_dim": 128,
            "routed_expert_temperature": 1.0,
            "routed_expert_scale": 0.5,
            "reliability_threshold": 0.15,
            "reliability_temperature": 0.1,
            "use_reliability_gate": True,
        }
        full_dara_defaults.update(model_kwargs)
        backbone_name = str(
            full_dara_defaults.pop(
                "backbone_name",
                "densenet121" if is_densenet_variant else "convnext_tiny",
            )
        )
        if normalized_name in legacy_qdhe_aliases:
            classifier_type = QDHEClassifier
        elif normalized_name in aeris_full_aliases:
            classifier_type = AERISClassifier
        else:
            classifier_type = DARAClassifier
        return classifier_type(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **full_dara_defaults,
        )
    if normalized_name in {"aeris_base", "aeris_base_t"}:
        model_kwargs = dict(model_kwargs)
        base_defaults = {
            "stage_proj_dims": [192, 256, 320],
            "stage_fusion_weights": [0.8, 1.0, 1.0],
            "head_hidden_dim": 320,
            "head_dropout": 0.2,
            "active_stage_indices": [2, 3, 4],
            "use_stage4_attention": False,
            "use_projection_grn": True,
            "use_projection_eca": True,
            "use_stage_local_mixer": True,
            "stage_local_mixer_stages": [2, 3],
            "use_scale_gate": False,
            "use_stage_attn_res": True,
            "use_delta_history": False,
            "use_shared_routed_experts": False,
            "use_reliability_gate": False,
        }
        base_defaults.update(model_kwargs)
        backbone_name = str(base_defaults.pop("backbone_name", "convnext_tiny"))
        return AERISClassifier(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **base_defaults,
        )
    if normalized_name in {"dara_tiny", "dara_moe_tiny", "dara_stage_moe_tiny", "dara"}:
        model_kwargs = dict(model_kwargs)
        dara_defaults = {
            "stage_proj_dims": [192, 256, 320],
            "stage_fusion_weights": [0.8, 1.0, 1.0],
            "head_hidden_dim": 320,
            "head_dropout": 0.2,
            "active_stage_indices": [2, 3, 4],
            "use_stage4_attention": False,
            "use_projection_grn": True,
            "use_projection_eca": True,
            "use_stage_local_mixer": True,
            "stage_local_mixer_stages": [2, 3],
            "use_scale_gate": False,
            "use_stage_attn_res": True,
            "stage_attn_res_mode": "content",
            "stage_attn_res_attention_dim": 128,
            "stage_attn_res_temperature": 1.0,
            "stage_attn_res_strength": 0.75,
            "use_shared_routed_experts": normalized_name != "dara_tiny",
            "shared_expert_hidden_dim": 256,
            "routed_expert_hidden_dim": 128,
            "routed_expert_count": 4,
            "routed_expert_top_k": 2,
            "routed_expert_router_hidden_dim": 128,
            "routed_expert_temperature": 1.0,
            "routed_expert_scale": 0.5,
        }
        dara_defaults.update(model_kwargs)
        backbone_name = str(dara_defaults.pop("backbone_name", "convnext_tiny"))
        return DARAClassifier(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **dara_defaults,
        )
    if normalized_name in {
        "lesionext_moe",
        "lesionext_moe_tiny",
        "lesionext_moe_v1_tiny",
    }:
        model_kwargs = dict(model_kwargs)
        moe_defaults = {
            "stage_proj_dims": [192, 256, 320],
            "stage_fusion_weights": [0.8, 1.0, 1.0],
            "attn_heads": 4,
            "attn_mlp_ratio": 2.0,
            "head_hidden_dim": 256,
            "attn_dropout": 0.1,
            "head_dropout": 0.2,
            "active_stage_indices": [2, 3, 4],
            "use_stage4_attention": False,
            "use_projection_grn": True,
            "use_projection_eca": True,
            "use_scale_gate": False,
            "descriptor_dim": 14,
            "expert_hidden_dim": 256,
            "router_hidden_dim": 160,
            "router_temperature": 1.0,
        }
        moe_defaults.update(model_kwargs)
        backbone_name = str(moe_defaults.pop("backbone_name", "convnext_tiny"))
        return LesionScaleMoEConvNeXtClassifier(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **moe_defaults,
        )
    if normalized_name in {
        "lesionext_moe_v2_tiny",
    }:
        model_kwargs = dict(model_kwargs)
        moe_defaults = {
            "stage_proj_dims": [192, 256, 320],
            "stage_fusion_weights": [0.8, 1.0, 1.0],
            "attn_heads": 4,
            "attn_mlp_ratio": 2.0,
            "head_hidden_dim": 256,
            "attn_dropout": 0.1,
            "head_dropout": 0.2,
            "active_stage_indices": [2, 3, 4],
            "use_stage4_attention": False,
            "use_projection_grn": True,
            "use_projection_eca": True,
            "use_scale_gate": False,
            "descriptor_dim": 14,
            "expert_hidden_dim": 256,
            "router_hidden_dim": 160,
            "router_temperature": 1.0,
            "area_prior_strength": 0.75,
            "area_prior_sigma": 0.22,
            "small_spatial_hidden_dim": 96,
            "residual_alpha": 0.15,
        }
        moe_defaults.update(model_kwargs)
        backbone_name = str(moe_defaults.pop("backbone_name", "convnext_tiny"))
        return LesionScaleMoEV2ConvNeXtClassifier(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **moe_defaults,
        )
    if normalized_name in {
        "lesionext_moe_v3_tiny",
        "sonoglore_lesion_moe_v3_convnext_tiny",
    }:
        model_kwargs = dict(model_kwargs)
        moe_defaults = {
            "stage_proj_dims": [192, 256, 320],
            "stage_fusion_weights": [0.8, 1.0, 1.0],
            "attn_heads": 4,
            "attn_mlp_ratio": 2.0,
            "head_hidden_dim": 256,
            "attn_dropout": 0.1,
            "head_dropout": 0.2,
            "active_stage_indices": [2, 3, 4],
            "use_stage4_attention": False,
            "use_projection_grn": True,
            "use_projection_eca": True,
            "use_scale_gate": False,
            "descriptor_dim": 14,
            "expert_hidden_dim": 256,
            "router_hidden_dim": 160,
            "router_temperature": 1.5,
            "area_prior_strength": 0.3,
            "area_prior_sigma": 0.22,
            "small_spatial_hidden_dim": 96,
            "residual_alpha": 0.15,
            "max_valid_area_ratio": 0.75,
            "expert_weight_floor": 0.05,
        }
        moe_defaults.update(model_kwargs)
        backbone_name = str(moe_defaults.pop("backbone_name", "convnext_tiny"))
        return LesionScaleMoEV3ConvNeXtClassifier(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **moe_defaults,
        )
    if normalized_name in {
        "lesionext_moe_v4_tiny",
        "lesionext_v4_tiny",
    }:
        model_kwargs = dict(model_kwargs)
        moe_defaults = {
            "stage_proj_dims": [192, 256, 320],
            "stage_fusion_weights": [0.8, 1.0, 1.0],
            "attn_heads": 4,
            "attn_mlp_ratio": 2.0,
            "head_hidden_dim": 256,
            "attn_dropout": 0.1,
            "head_dropout": 0.2,
            "active_stage_indices": [2, 3, 4],
            "use_stage4_attention": False,
            "use_projection_grn": True,
            "use_projection_eca": True,
            "use_scale_gate": False,
            "descriptor_dim": 14,
            "expert_hidden_dim": 256,
            "router_hidden_dim": 160,
            "router_temperature": 1.5,
            "area_prior_strength": 0.3,
            "area_prior_sigma": 0.22,
            "small_spatial_hidden_dim": 96,
            "residual_alpha": 0.15,
            "max_valid_area_ratio": 0.75,
            "expert_weight_floor": 0.05,
            "use_domain_invariant_norm": True,
            "domain_norm_stages": [2, 3, 4],
        }
        moe_defaults.update(model_kwargs)
        backbone_name = str(moe_defaults.pop("backbone_name", "convnext_tiny"))
        return LesionScaleMoEV3ConvNeXtClassifier(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **moe_defaults,
        )
    if normalized_name == "lesionext_moe_v3_1_tiny":
        model_kwargs = dict(model_kwargs)
        moe_defaults = {
            "stage_proj_dims": [192, 256, 320],
            "stage_fusion_weights": [0.8, 1.0, 1.0],
            "attn_heads": 4,
            "attn_mlp_ratio": 2.0,
            "head_hidden_dim": 256,
            "attn_dropout": 0.1,
            "head_dropout": 0.2,
            "active_stage_indices": [2, 3, 4],
            "use_stage4_attention": False,
            "use_projection_grn": True,
            "use_projection_eca": True,
            "use_scale_gate": False,
            "descriptor_dim": 14,
            "expert_hidden_dim": 256,
            "router_hidden_dim": 160,
            "router_temperature": 1.2,
            "area_prior_strength": 0.35,
            "area_prior_sigma": 0.22,
            "small_spatial_hidden_dim": 96,
            "residual_alpha": 0.15,
            "max_valid_area_ratio": 0.75,
            "expert_weight_floor": 0.03,
        }
        moe_defaults.update(model_kwargs)
        backbone_name = str(moe_defaults.pop("backbone_name", "convnext_tiny"))
        return LesionScaleMoEV3ConvNeXtClassifier(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **moe_defaults,
        )
    if normalized_name in {
        "lesionext_tiny",
        "lesionext_small",
        "lesionext",
        "lesionext_v1",
    }:
        model_kwargs = dict(model_kwargs)
        if normalized_name in {
            "lesionext",
            "lesionext_v1",
        }:
            default_backbone = "convnext_tiny"
            v1_defaults = {
                "proj_dim": 256,
                "stage_proj_dims": [192, 256, 320],
                "stage_fusion_weights": [0.8, 1.0, 1.0],
                "attn_heads": 4,
                "attn_mlp_ratio": 2.0,
                "head_hidden_dim": 256,
                "attn_dropout": 0.1,
                "head_dropout": 0.2,
                "active_stage_indices": [2, 3, 4],
                "use_stage4_attention": False,
                "use_projection_grn": True,
                "use_projection_eca": True,
                "use_scale_gate": False,
            }
            v1_defaults.update(model_kwargs)
            model_kwargs = v1_defaults
        else:
            default_backbone = "convnext_tiny" if normalized_name.endswith("tiny") else "convnext_small"
        backbone_name = str(model_kwargs.pop("backbone_name", default_backbone))
        return LesioNeXtClassifier(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **model_kwargs,
        )
    if normalized_name in {"roi_dualview_convnext_tiny", "dualview_convnext_tiny"}:
        backbone_name = str(model_kwargs.pop("backbone_name", "convnext_tiny"))
        return DualViewConvNeXtClassifier(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **model_kwargs,
        )
    if normalized_name == "alexnet" and torchvision_models is not None:
        weights = torchvision_models.AlexNet_Weights.DEFAULT if pretrained else None
        model = torchvision_models.alexnet(weights=weights)
        if in_chans != 3:
            model.features[0] = nn.Conv2d(
                in_chans,
                model.features[0].out_channels,
                kernel_size=model.features[0].kernel_size,
                stride=model.features[0].stride,
                padding=model.features[0].padding,
            )
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if timm is not None:
        return timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
            **model_kwargs,
        )
    return TinyCNNClassifier(in_chans=in_chans, num_classes=num_classes)


def load_classifier(
    model_config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    *,
    map_location: str = "cpu",
):
    """Create a classifier and load a checkpoint when the path exists."""
    extra_model_kwargs = {}
    nested_model_kwargs = model_config.get("model_kwargs")
    if isinstance(nested_model_kwargs, dict):
        extra_model_kwargs.update(nested_model_kwargs)
    for key, value in model_config.items():
        if key in {"name", "pretrained", "in_chans", "num_classes", "model_kwargs"}:
            continue
        extra_model_kwargs[key] = value
    model = create_classifier(
        model_name=model_config.get("name", "resnet18"),
        pretrained=bool(model_config.get("pretrained", False)),
        in_chans=int(model_config.get("in_chans", 3)),
        num_classes=int(model_config.get("num_classes", 2)),
        **extra_model_kwargs,
    )
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=map_location)
        state_dict = state.get("state_dict", state.get("model", state))
        model.load_state_dict(state_dict, strict=False)
    return model


def classifier_probabilities(model, batch, *, device: str = "cpu", move_model: bool = True):
    """Return softmax class probabilities for one batch."""
    require_dependency("torch", torch)
    require_dependency("torch.nn.functional", F)
    if move_model:
        model = model.to(device)
    if batch.ndim == 3:
        batch = batch.unsqueeze(0)
    batch = batch.to(device=device, dtype=torch.float32)
    inference_context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
    with inference_context():
        logits = model(batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    return probs


def resolve_gradcam_target_layer(model) -> Any | None:
    """Find a reasonable final feature layer for Grad-CAM on common model families."""
    explicit_layer = getattr(model, "gradcam_layer", None)
    if explicit_layer is not None:
        return explicit_layer
    target_layer_getter = getattr(model, "get_gradcam_target_layer", None)
    if callable(target_layer_getter):
        target_layer = target_layer_getter()
        if target_layer is not None:
            return target_layer
    for candidate in ("layer4", "features", "stages", "blocks", "conv_head", "backbone"):
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

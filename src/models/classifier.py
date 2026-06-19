"""Classifier model factory, checkpoint loader, and probability helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from src.utils.runtime import optional_import, require_dependency


torch = optional_import("torch")
nn = optional_import("torch.nn")
F = optional_import("torch.nn.functional")
timm = optional_import("timm")
torchvision_models = optional_import("torchvision.models")


if nn is not None:
    VALID_SONOGLORE_STAGE_INDEX_GROUPS = {
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


    class SonoGloReNetClassifier(nn.Module):
        """ConvNeXt-based classifier with multi-scale pooling and lightweight attention."""

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
            learnable_stage_fusion: bool = False,
            fusion_weight_epsilon: float = 1e-4,
            classifier_head_type: str = "linear",
            classifier_head_scale: float = 16.0,
            classifier_head_margin: float = 0.15,
            classifier_head_easy_margin: bool = False,
            **backbone_kwargs,
        ) -> None:
            """Build SonoGloReNet from a timm ConvNeXt backbone."""
            super().__init__()
            if timm is None:
                raise RuntimeError("timm is required to construct SonoGloReNet.")
            self.backbone_name = str(backbone_name)
            self.use_stage4_attention = bool(use_stage4_attention)
            self.use_projection_grn = bool(use_projection_grn)
            self.use_projection_eca = bool(use_projection_eca)
            self.use_projection_spectral_gate = bool(use_projection_spectral_gate)
            self.use_mixstyle = bool(use_mixstyle)
            self.use_stage_local_mixer = bool(use_stage_local_mixer)
            self.use_scale_gate = bool(use_scale_gate)
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
                    f"SonoGloReNet expects three feature stages, got {len(channels)} from {self.backbone_name!r}."
                )
            self._stage_numbers = (2, 3, 4)
            self._stage_number_to_position = {
                stage_number: position
                for position, stage_number in enumerate(self._stage_numbers)
            }
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
            self.last_scale_gate_weights = None
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
            if normalized not in VALID_SONOGLORE_STAGE_INDEX_GROUPS:
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

        def _fused_embedding(self, x):
            """Return the fused representation before classifier-only dropout."""
            stage_features = list(self.backbone(x))
            if len(stage_features) != 3:
                raise RuntimeError(f"SonoGloReNet received {len(stage_features)} features, expected 3.")
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
                pooled_embeddings.append(pooled * fusion_factors[active_position])
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
            fused = torch.cat(pooled_embeddings, dim=1)
            fused = self.head_norm(fused)
            fused = self.head_dropout(fused)
            fused = self.head_fc1(fused)
            fused = self.head_act(fused)
            return fused

        def forward_with_embedding(self, x, labels=None):
            """Return class logits together with the fused training embedding."""
            fused = self._fused_embedding(x)
            classifier_input = self.head_hidden_dropout(fused)
            if self.classifier_head_type == "arc_margin":
                logits = self.classifier(classifier_input, labels=labels)
            else:
                logits = self.classifier(classifier_input)
            return logits, fused

        def forward(self, x, labels=None):
            """Return class logits for one image batch."""
            logits, _embedding = self.forward_with_embedding(x, labels=labels)
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


    class SonoGloReNetClassifier:  # type: ignore[override]
        """Placeholder SonoGloReNet used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct SonoGloReNet.")


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
    if normalized_name in {"sonoglore_convnext_tiny", "sonoglore_convnext_small"}:
        default_backbone = "convnext_tiny" if normalized_name.endswith("tiny") else "convnext_small"
        backbone_name = str(model_kwargs.pop("backbone_name", default_backbone))
        return SonoGloReNetClassifier(
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

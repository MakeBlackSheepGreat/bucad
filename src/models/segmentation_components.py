"""Custom segmentation blocks used by paper-guided segmenter variants."""

from __future__ import annotations

import math
from typing import Any

from src.models.dara import (
    CrossStageDeltaHistory2d,
    ReliabilityGatedSpatialExpertHead,
    SharedRoutedSpatialMoE,
    StageAttentionResidualFusion,
)
from src.utils.runtime import optional_import, require_dependency


torch = optional_import("torch")
nn = optional_import("torch.nn")
F = optional_import("torch.nn.functional")
timm = optional_import("timm")


if nn is not None:
    class ConvNormAct(nn.Module):
        """Small Conv-BatchNorm-ReLU block reused by custom segmenters."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            kernel_size: int = 3,
            groups: int = 1,
        ) -> None:
            """Build a convolution, batch norm, and SiLU activation block."""
            super().__init__()
            padding = kernel_size // 2
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            """Apply the convolutional block."""
            return self.block(x)


    class FeatureEdgeAmplifier(nn.Module):
        """CENet-style multi-scale difference block for skip edge enhancement."""

        def __init__(self, channels: int, *, scales: tuple[int, ...] = (2, 4)) -> None:
            """Create pooled-difference branches for skip feature sharpening."""
            super().__init__()
            self.scales = scales
            self.fuse = ConvNormAct(channels * (len(scales) + 1), channels, kernel_size=1)

        def forward(self, x):
            """Concatenate original and pooled-difference features, then fuse."""
            require_dependency("torch.nn.functional", F)
            features = [x]
            for scale in self.scales:
                pooled = F.avg_pool2d(x, kernel_size=scale, stride=scale, ceil_mode=True)
                upsampled = F.interpolate(
                    pooled,
                    size=x.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                features.append(x - upsampled)
            return self.fuse(torch.cat(features, dim=1))


    class DifferentialAttention(nn.Module):
        """Subtracts a coarse context attention map from an edge attention map."""

        def __init__(self, channels: int) -> None:
            """Create edge and context attention branches."""
            super().__init__()
            hidden = max(8, channels // 4)
            self.edge_attention = nn.Sequential(
                ConvNormAct(channels, hidden, kernel_size=3),
                nn.Conv2d(hidden, 1, kernel_size=1),
            )
            self.context_attention = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                ConvNormAct(channels, hidden, kernel_size=3),
                nn.Conv2d(hidden, 1, kernel_size=1),
            )

        def forward(self, x):
            """Apply differential attention to emphasize likely edge features."""
            attention = torch.sigmoid(self.edge_attention(x) - self.context_attention(x))
            return x * (1.0 + attention)


    class DSEB(nn.Module):
        """Differential skip enhancement block inspired by CENet."""

        def __init__(self, channels: int) -> None:
            """Create edge amplification, differential attention, and output fusion."""
            super().__init__()
            self.edge = FeatureEdgeAmplifier(channels)
            self.attention = DifferentialAttention(channels)
            self.output = ConvNormAct(channels, channels, kernel_size=3)

        def forward(self, x):
            """Enhance skip features and preserve the residual signal."""
            enhanced = self.attention(self.edge(x))
            return self.output(x + enhanced)


    class ChannelCalibrationUnit(nn.Module):
        """Channel attention block using mean, max, and standard deviation descriptors."""

        def __init__(self, channels: int, *, reduction: int = 8) -> None:
            """Build the channel calibration MLP."""
            super().__init__()
            hidden = max(4, channels // reduction)
            self.mlp = nn.Sequential(
                nn.Conv2d(channels * 3, hidden, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1),
            )

        def forward(self, x):
            """Apply channel-wise calibration weights."""
            avg = x.mean(dim=(2, 3), keepdim=True)
            max_values = x.amax(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True, unbiased=False)
            weights = torch.sigmoid(self.mlp(torch.cat([avg, max_values, std], dim=1)))
            return x * (1.0 + weights)


    class MultiScaleContextAggregator(nn.Module):
        """Depthwise multi-kernel context aggregator."""

        def __init__(self, channels: int, *, kernels: tuple[int, ...] = (3, 5, 7)) -> None:
            """Build one depthwise branch per kernel size and a fusion block."""
            super().__init__()
            branches = []
            for kernel in kernels:
                branches.append(
                    nn.Sequential(
                        ConvNormAct(channels, channels, kernel_size=kernel, groups=channels),
                        ConvNormAct(channels, channels, kernel_size=1),
                    )
                )
            self.branches = nn.ModuleList(branches)
            self.fuse = ConvNormAct(channels * len(kernels), channels, kernel_size=1)

        def forward(self, x):
            """Fuse context branches along the channel dimension."""
            return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


    class WeightedNonLocalLite(nn.Module):
        """Lightweight non-local context block with learnable residual weight."""

        def __init__(self, channels: int, *, pool_size: int = 16) -> None:
            """Build query/key/value projections over pooled spatial features."""
            super().__init__()
            hidden = max(8, channels // 2)
            self.pool_size = pool_size
            self.query = nn.Conv2d(channels, hidden, kernel_size=1)
            self.key = nn.Conv2d(channels, hidden, kernel_size=1)
            self.value = nn.Conv2d(channels, channels, kernel_size=1)
            self.weight = nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
            """Apply pooled self-attention and add it back as a residual context."""
            require_dependency("torch.nn.functional", F)
            batch, _, height, width = x.shape
            pooled = F.adaptive_avg_pool2d(x, output_size=(self.pool_size, self.pool_size))
            query = self.query(pooled).flatten(2).transpose(1, 2)
            key = self.key(pooled).flatten(2)
            value = self.value(pooled).flatten(2).transpose(1, 2)
            scale = max(1, key.shape[1]) ** -0.5
            attention = torch.softmax(torch.bmm(query, key) * scale, dim=-1)
            context = torch.bmm(attention, value).transpose(1, 2)
            context = context.reshape(batch, -1, self.pool_size, self.pool_size)
            context = F.interpolate(context, size=(height, width), mode="bilinear", align_corners=False)
            return x + torch.tanh(self.weight) * context


    class CFAM(nn.Module):
        """CENet-style channel calibration and multi-scale context fusion."""

        def __init__(self, channels: int, *, use_nonlocal: bool = True) -> None:
            """Create channel calibration, context aggregation, and optional non-local block."""
            super().__init__()
            self.channel = ChannelCalibrationUnit(channels)
            self.context = MultiScaleContextAggregator(channels)
            self.nonlocal_block = WeightedNonLocalLite(channels) if use_nonlocal else nn.Identity()
            self.output = ConvNormAct(channels, channels, kernel_size=3)

        def forward(self, x):
            """Fuse calibrated context with the original residual feature."""
            calibrated = self.channel(x)
            contextual = self.context(calibrated)
            return self.output(self.nonlocal_block(contextual) + x)


    class EncoderStage(nn.Module):
        """Two-convolution encoder stage with optional downsampling."""

        def __init__(self, in_channels: int, out_channels: int, *, downsample: bool) -> None:
            """Build one encoder stage."""
            super().__init__()
            layers = []
            if downsample:
                layers.append(nn.MaxPool2d(2))
            layers.extend(
                [
                    ConvNormAct(in_channels, out_channels),
                    ConvNormAct(out_channels, out_channels),
                ]
            )
            self.block = nn.Sequential(*layers)

        def forward(self, x):
            """Return encoded features for this stage."""
            return self.block(x)


    class DecoderStage(nn.Module):
        """Decoder stage that upsamples, enhances skip features, and fuses context."""

        def __init__(
            self,
            in_channels: int,
            skip_channels: int,
            out_channels: int,
            *,
            use_dseb: bool,
            use_cfam: bool,
            use_nonlocal: bool,
        ) -> None:
            """Build one decoder stage with optional DSEB and CFAM."""
            super().__init__()
            self.skip_enhance = DSEB(skip_channels) if use_dseb else nn.Identity()
            self.reduce = ConvNormAct(in_channels + skip_channels, out_channels)
            self.context = CFAM(out_channels, use_nonlocal=use_nonlocal) if use_cfam else nn.Identity()
            self.output = ConvNormAct(out_channels, out_channels)

        def forward(self, x, skip):
            """Upsample decoder features, fuse with skip features, and return refined output."""
            require_dependency("torch.nn.functional", F)
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            skip = self.skip_enhance(skip)
            x = self.reduce(torch.cat([x, skip], dim=1))
            return self.output(self.context(x))


    class CENetLite(nn.Module):
        """Compact CE-Net style decoder used for lightweight segmentation trials."""

        def __init__(
            self,
            *,
            in_channels: int = 3,
            classes: int = 1,
            base_channels: int = 32,
            use_dseb: bool = True,
            use_cfam: bool = True,
            use_nonlocal: bool = True,
            boundary_head: bool = False,
        ) -> None:
            """Assemble encoder, context block, decoder, mask head, and optional boundary head."""
            super().__init__()
            widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
            self.boundary_head_enabled = bool(boundary_head)
            self.enc1 = EncoderStage(in_channels, widths[0], downsample=False)
            self.enc2 = EncoderStage(widths[0], widths[1], downsample=True)
            self.enc3 = EncoderStage(widths[1], widths[2], downsample=True)
            self.enc4 = EncoderStage(widths[2], widths[3], downsample=True)
            self.bottleneck = CFAM(widths[3], use_nonlocal=use_nonlocal) if use_cfam else ConvNormAct(widths[3], widths[3])
            self.dec3 = DecoderStage(
                widths[3],
                widths[2],
                widths[2],
                use_dseb=use_dseb,
                use_cfam=use_cfam,
                use_nonlocal=use_nonlocal,
            )
            self.dec2 = DecoderStage(
                widths[2],
                widths[1],
                widths[1],
                use_dseb=use_dseb,
                use_cfam=use_cfam,
                use_nonlocal=use_nonlocal,
            )
            self.dec1 = DecoderStage(
                widths[1],
                widths[0],
                widths[0],
                use_dseb=use_dseb,
                use_cfam=use_cfam,
                use_nonlocal=use_nonlocal,
            )
            self.mask_head = nn.Conv2d(widths[0], classes, kernel_size=1)
            self.boundary_head = nn.Conv2d(widths[0], classes, kernel_size=1) if boundary_head else None

        def _forward_features(self, x) -> dict[str, Any]:
            """Return decoder and bottleneck features before task heads."""
            s1 = self.enc1(x)
            s2 = self.enc2(s1)
            s3 = self.enc3(s2)
            s4 = self.enc4(s3)
            x = self.bottleneck(s4)
            x = self.dec3(x, s3)
            x = self.dec2(x, s2)
            decoder_feature = self.dec1(x, s1)
            return {"decoder_feature": decoder_feature, "bottleneck": s4}

        def forward_with_aux(self, x) -> dict[str, Any]:
            """Return mask logits plus auxiliary features and optional boundary logits."""
            features = self._forward_features(x)
            mask = self.mask_head(features["decoder_feature"])
            output = {
                "mask": mask,
                "prototype_features": features["decoder_feature"],
                "bottleneck": features["bottleneck"],
            }
            if self.boundary_head is not None:
                output["boundary"] = self.boundary_head(features["decoder_feature"])
            return output

        def forward(self, x):
            """Return only mask logits for standard segmentation callers."""
            return self.forward_with_aux(x)["mask"]


    class TimmFeaturePyramidSegmenter(nn.Module):
        """Feature-pyramid segmenter backed by a timm encoder for replacement trials."""

        def __init__(
            self,
            *,
            encoder_name: str = "pvt_v2_b0",
            encoder_weights: str | None = None,
            in_channels: int = 3,
            classes: int = 1,
            decoder_channels: int = 128,
            use_dseb: bool = True,
            use_cfam: bool = True,
            use_nonlocal: bool = True,
            boundary_head: bool = False,
        ) -> None:
            """Build a timm feature pyramid decoder with optional boundary output."""
            super().__init__()
            if timm is None:
                raise RuntimeError("timm is required for the PVT-v2 CENet experiment.")
            pretrained = encoder_weights not in {None, "none", "None", False}
            try:
                self.encoder = timm.create_model(
                    encoder_name,
                    pretrained=bool(pretrained),
                    features_only=True,
                    in_chans=in_channels,
                    out_indices=(0, 1, 2, 3),
                )
            except Exception as exc:
                raise RuntimeError(f"Unable to construct timm encoder {encoder_name!r}: {exc}") from exc
            channels = list(self.encoder.feature_info.channels())
            self.boundary_head_enabled = bool(boundary_head)
            self.projections = nn.ModuleList(
                [ConvNormAct(channel, decoder_channels, kernel_size=1) for channel in channels]
            )
            self.skip_enhance = nn.ModuleList(
                [DSEB(decoder_channels) if use_dseb else nn.Identity() for _ in channels[:-1]]
            )
            self.context = CFAM(decoder_channels, use_nonlocal=use_nonlocal) if use_cfam else nn.Identity()
            self.output = ConvNormAct(decoder_channels, decoder_channels)
            self.mask_head = nn.Conv2d(decoder_channels, classes, kernel_size=1)
            self.boundary_head = nn.Conv2d(decoder_channels, classes, kernel_size=1) if boundary_head else None

        def forward_with_aux(self, x) -> dict[str, Any]:
            """Run the timm feature pyramid and return mask plus auxiliary outputs."""
            require_dependency("torch.nn.functional", F)
            features = self.encoder(x)
            projected = [projection(feature) for projection, feature in zip(self.projections, features)]
            decoder = projected[-1]
            for index in range(len(projected) - 2, -1, -1):
                skip = self.skip_enhance[index](projected[index])
                decoder = F.interpolate(
                    decoder,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                decoder = decoder + skip
                decoder = self.context(decoder)
            decoder = self.output(decoder)
            decoder = F.interpolate(decoder, size=x.shape[-2:], mode="bilinear", align_corners=False)
            mask = self.mask_head(decoder)
            output = {"mask": mask, "prototype_features": decoder}
            if self.boundary_head is not None:
                output["boundary"] = self.boundary_head(decoder)
            return output

        def forward(self, x):
            """Return only mask logits for standard segmentation callers."""
            return self.forward_with_aux(x)["mask"]


    class DARAFeaturePyramidSegmenter(nn.Module):
        """Historical dense model retained for DARA/QDHE checkpoint compatibility."""

        def __init__(
            self,
            *,
            encoder_name: str = "convnext_tiny",
            encoder_weights: str | None = "imagenet",
            in_channels: int = 3,
            classes: int = 1,
            decoder_channels: int = 128,
            use_dseb: bool = True,
            use_cfam: bool = True,
            use_nonlocal: bool = True,
            boundary_head: bool = True,
            stage_attn_res_attention_dim: int = 128,
            stage_attn_res_temperature: float = 1.0,
            stage_attn_res_strength: float = 0.75,
            fusion_mode: str = "stage_attn",
            delta_history_attention_dim: int = 64,
            delta_history_temperature: float = 0.8,
            delta_history_strength: float = 0.5,
            use_spatial_moe: bool = True,
            spatial_moe_expert_count: int = 4,
            spatial_moe_top_k: int = 2,
            spatial_moe_dilations: list[int] | tuple[int, ...] = (1, 2, 3, 5),
            spatial_moe_temperature: float = 1.0,
            spatial_moe_scale: float = 0.5,
            spatial_moe_shared_scale: float = 0.1,
            spatial_moe_mode: str = "feature",
            use_reliability_gate: bool = False,
            reliability_threshold: float = 0.15,
            reliability_temperature: float = 0.1,
            adaptive_compute: bool = False,
            min_active_experts: int = 0,
            benefit_hidden_dim: int = 64,
            benefit_thresholds: list[float] | tuple[float, ...] = (0.0, 0.0),
            gsd_adaptive_compute: bool = False,
            gsd_reference_meters: float = 1.0,
            gsd_compute_gain: float = 0.25,
            source_free_adaptation: bool = False,
            source_free_momentum: float = 0.9,
            source_free_min_reliability: float = 0.6,
            source_free_strength: float = 0.05,
            spatial_allocation: bool = False,
            spatial_allocation_strength: float = 0.5,
        ) -> None:
            """Build a reusable DARA encoder-decoder for dense prediction."""
            super().__init__()
            if timm is None:
                raise RuntimeError("timm is required for DARAFeaturePyramidSegmenter.")
            pretrained = encoder_weights not in {None, "none", "None", False}
            self.encoder = timm.create_model(
                encoder_name,
                pretrained=bool(pretrained),
                features_only=True,
                in_chans=in_channels,
                out_indices=(0, 1, 2, 3),
            )
            channels = list(self.encoder.feature_info.channels())
            if len(channels) != 4:
                raise RuntimeError(f"DARA segmenter expected four encoder stages, got {len(channels)}.")
            decoder_channels = int(decoder_channels)
            self.boundary_head_enabled = bool(boundary_head)
            self.projections = nn.ModuleList(
                [ConvNormAct(channel, decoder_channels, kernel_size=1) for channel in channels]
            )
            resolved_fusion_mode = str(fusion_mode).lower()
            if resolved_fusion_mode not in {
                "fixed",
                "stage_attn",
                "delta_history",
                "hybrid",
                "quality_history",
            }:
                raise ValueError(
                    "fusion_mode must be fixed, stage_attn, delta_history, hybrid, or quality_history."
                )
            self.fusion_mode = resolved_fusion_mode
            with torch.random.fork_rng(devices=[]):
                self.stage_attn_res = (
                    StageAttentionResidualFusion(
                        [decoder_channels] * len(channels),
                        attention_dim=int(stage_attn_res_attention_dim),
                        temperature=float(stage_attn_res_temperature),
                        residual_strength=float(stage_attn_res_strength),
                        query_mode="content",
                        initial_weights=(0.7, 0.85, 1.0, 1.0),
                    )
                    if self.fusion_mode in {"stage_attn", "hybrid", "quality_history"}
                    else None
                )
            with torch.random.fork_rng(devices=[]):
                self.delta_history = (
                    CrossStageDeltaHistory2d(
                        decoder_channels,
                        len(channels),
                        attention_dim=int(delta_history_attention_dim),
                        temperature=float(delta_history_temperature),
                        residual_strength=float(delta_history_strength),
                    )
                    if self.fusion_mode in {"delta_history", "hybrid", "quality_history"}
                    else None
                )
            self.hybrid_delta_gates = (
                nn.Parameter(torch.full((len(channels),), 0.1))
                if self.fusion_mode == "hybrid"
                else None
            )
            self.skip_enhance = nn.ModuleList(
                [DSEB(decoder_channels) if use_dseb else nn.Identity() for _ in channels[:-1]]
            )
            self.fusion_blocks = nn.ModuleList(
                [ConvNormAct(decoder_channels, decoder_channels) for _ in channels[:-1]]
            )
            self.context = CFAM(decoder_channels, use_nonlocal=use_nonlocal) if use_cfam else nn.Identity()
            self.spatial_moe_mode = str(spatial_moe_mode).lower()
            if self.spatial_moe_mode not in {"feature", "logit"}:
                raise ValueError("spatial_moe_mode must be feature or logit.")
            with torch.random.fork_rng(devices=[]):
                self.spatial_moe = (
                    SharedRoutedSpatialMoE(
                        decoder_channels,
                        num_routed_experts=int(spatial_moe_expert_count),
                        top_k=int(spatial_moe_top_k),
                        dilations=spatial_moe_dilations,
                        router_temperature=float(spatial_moe_temperature),
                        routed_scale=float(spatial_moe_scale),
                        shared_residual_scale=float(spatial_moe_shared_scale),
                        use_reliability_gate=bool(use_reliability_gate),
                        reliability_threshold=float(reliability_threshold),
                        reliability_temperature=float(reliability_temperature),
                        adaptive_compute=bool(adaptive_compute),
                        min_active_experts=int(min_active_experts),
                        benefit_hidden_dim=int(benefit_hidden_dim),
                        benefit_thresholds=benefit_thresholds,
                        gsd_adaptive_compute=bool(gsd_adaptive_compute),
                        gsd_reference_meters=float(gsd_reference_meters),
                        gsd_compute_gain=float(gsd_compute_gain),
                        source_free_adaptation=bool(source_free_adaptation),
                        source_free_momentum=float(source_free_momentum),
                        source_free_min_reliability=float(source_free_min_reliability),
                        source_free_strength=float(source_free_strength),
                        spatial_allocation=bool(spatial_allocation),
                        spatial_allocation_strength=float(spatial_allocation_strength),
                    )
                    if use_spatial_moe and self.spatial_moe_mode == "feature"
                    else nn.Identity()
                )
            self.output = ConvNormAct(decoder_channels, decoder_channels)
            with torch.random.fork_rng(devices=[]):
                self.logit_moe = (
                    ReliabilityGatedSpatialExpertHead(
                        decoder_channels,
                        classes,
                        num_routed_experts=int(spatial_moe_expert_count),
                        top_k=int(spatial_moe_top_k),
                        dilations=spatial_moe_dilations,
                        router_temperature=float(spatial_moe_temperature),
                        routed_scale=float(spatial_moe_scale),
                        use_reliability_gate=bool(use_reliability_gate),
                        reliability_threshold=float(reliability_threshold),
                        reliability_temperature=float(reliability_temperature),
                    )
                    if use_spatial_moe and self.spatial_moe_mode == "logit"
                    else None
                )
            self.mask_head = (
                nn.Conv2d(decoder_channels, classes, kernel_size=1)
                if self.logit_moe is None
                else None
            )
            self.boundary_head = nn.Conv2d(decoder_channels, classes, kernel_size=1) if boundary_head else None
            self.last_stage_attn_res_weights = None
            self.last_delta_history_weights = None
            self.last_dense_expert_weights = None
            self.last_expert_weights = None
            self.last_stage_selection_confidence = None
            self.last_delta_history_confidence = None
            self.last_depth_reliability = None
            self.last_routing_confidence = None
            self.last_combined_reliability = None
            self.last_reliability_gate = None
            self.last_active_expert_count = None
            self.last_predicted_expert_benefits = None
            self.last_gsd_risk = None
            self.last_gsd_risk_logits = None
            self.last_spatial_priority_logits = None
            self.last_spatial_allocation = None
            self.last_spatial_active_fraction = None
            self.last_spatial_allocation_mean = None
            self.last_source_free_update_count = None
            self.supports_gsd_metadata = True

        def forward_with_aux(self, x, *, gsd=None) -> dict[str, Any]:
            """Return mask, boundary, prototype, and routing diagnostics."""
            require_dependency("torch.nn.functional", F)
            features = self.encoder(x)
            projected = [projection(feature) for projection, feature in zip(self.projections, features)]
            base_projected = projected
            stage_weights = None
            history_weights = None
            if self.stage_attn_res is not None:
                tokens = [F.adaptive_avg_pool2d(feature, 1).flatten(1) for feature in projected]
                _weighted_tokens, stage_weights = self.stage_attn_res(tokens)
                scales = 1.0 + self.stage_attn_res.residual_strength * (stage_weights - 1.0)
                projected = [
                    feature * scales[:, index : index + 1, None, None]
                    for index, feature in enumerate(projected)
                ]
            if self.delta_history is not None:
                delta_input = (
                    base_projected
                    if self.fusion_mode in {"hybrid", "quality_history"}
                    else projected
                )
                delta_projected, history_weights = self.delta_history(delta_input)
                if self.fusion_mode == "hybrid":
                    hybrid_scales = torch.tanh(self.hybrid_delta_gates)
                    projected = [
                        stage_feature
                        + hybrid_scales[index] * (delta_feature - base_feature)
                        for index, (stage_feature, delta_feature, base_feature) in enumerate(
                            zip(projected, delta_projected, base_projected)
                        )
                    ]
                elif self.fusion_mode != "quality_history":
                    projected = delta_projected
            decoder = projected[-1]
            for index in range(len(projected) - 2, -1, -1):
                skip = self.skip_enhance[index](projected[index])
                decoder = F.interpolate(
                    decoder,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                decoder = self.fusion_blocks[index](decoder + skip)
            decoder = self.context(decoder)
            stage_selection_confidence = None
            delta_history_confidence = None
            reliability_terms = []
            if history_weights is not None:
                deepest = history_weights[:, -1, :].clamp_min(1e-8)
                entropy = -(deepest * deepest.log()).sum(dim=1)
                delta_history_confidence = (
                    1.0 - entropy / max(math.log(float(deepest.shape[1])), 1e-8)
                )
                reliability_terms.append(delta_history_confidence)
            if stage_weights is not None:
                normalized_stage_weights = stage_weights / stage_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
                entropy = -(
                    normalized_stage_weights.clamp_min(1e-8)
                    * normalized_stage_weights.clamp_min(1e-8).log()
                ).sum(dim=1)
                stage_selection_confidence = (
                    1.0
                    - entropy
                    / max(math.log(float(normalized_stage_weights.shape[1])), 1e-8)
                )
                reliability_terms.append(stage_selection_confidence)
            depth_reliability = (
                torch.stack(reliability_terms, dim=0).mean(dim=0)
                if reliability_terms
                else None
            )
            moe_result = None
            if isinstance(self.spatial_moe, SharedRoutedSpatialMoE):
                moe_result = self.spatial_moe(
                    decoder,
                    external_reliability=depth_reliability,
                    gsd=gsd,
                    return_counterfactuals=bool(
                        self.training and self.spatial_moe.adaptive_compute
                    ),
                )
                decoder = (
                    moe_result["output"]
                    if isinstance(moe_result, dict)
                    else moe_result
                )
            else:
                decoder = self.spatial_moe(decoder)
            counterfactual_masks = None
            predicted_benefits = None
            gsd_risk = None
            gsd_risk_logits = None
            spatial_priority_logits = None
            spatial_allocation = None
            if isinstance(moe_result, dict):
                counterfactual_decoders = [
                    self.output(feature)
                    for feature in moe_result["counterfactual_features"]
                ]
                decoder = counterfactual_decoders[-1]
                counterfactual_masks = [
                    self.mask_head(
                        F.interpolate(
                            feature,
                            size=x.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    )
                    for feature in counterfactual_decoders
                ]
                predicted_benefits = moe_result.get("predicted_benefits")
                gsd_risk = moe_result.get("gsd_risk")
                gsd_risk_logits = moe_result.get("gsd_risk_logits")
                spatial_priority_logits = moe_result.get("spatial_priority_logits")
                spatial_allocation = moe_result.get("spatial_allocation")
            else:
                decoder = self.output(decoder)
            decoder = F.interpolate(decoder, size=x.shape[-2:], mode="bilinear", align_corners=False)
            mask = (
                self.logit_moe(decoder, external_reliability=depth_reliability)
                if self.logit_moe is not None
                else self.mask_head(decoder)
            )
            self.last_stage_attn_res_weights = stage_weights.detach() if stage_weights is not None else None
            self.last_delta_history_weights = history_weights.detach() if history_weights is not None else None
            routing_module = self.logit_moe if self.logit_moe is not None else self.spatial_moe
            self.last_dense_expert_weights = getattr(
                routing_module,
                "last_dense_expert_weights",
                None,
            )
            self.last_expert_weights = getattr(routing_module, "last_expert_weights", None)
            self.last_stage_selection_confidence = (
                stage_selection_confidence.detach()
                if stage_selection_confidence is not None
                else None
            )
            self.last_delta_history_confidence = (
                delta_history_confidence.detach()
                if delta_history_confidence is not None
                else None
            )
            self.last_depth_reliability = (
                depth_reliability.detach() if depth_reliability is not None else None
            )
            self.last_routing_confidence = getattr(
                routing_module,
                "last_routing_confidence",
                None,
            )
            self.last_combined_reliability = getattr(
                routing_module,
                "last_combined_reliability",
                None,
            )
            self.last_reliability_gate = getattr(routing_module, "last_reliability_gate", None)
            self.last_active_expert_count = getattr(
                routing_module,
                "last_active_expert_count",
                None,
            )
            self.last_predicted_expert_benefits = getattr(
                routing_module,
                "last_predicted_expert_benefits",
                None,
            )
            self.last_gsd_risk = getattr(routing_module, "last_gsd_risk", None)
            self.last_gsd_risk_logits = getattr(routing_module, "last_gsd_risk_logits", None)
            self.last_spatial_priority_logits = getattr(
                routing_module,
                "last_spatial_priority_logits",
                None,
            )
            self.last_spatial_allocation = getattr(routing_module, "last_spatial_allocation", None)
            self.last_spatial_active_fraction = getattr(
                routing_module,
                "last_spatial_active_fraction",
                None,
            )
            self.last_spatial_allocation_mean = getattr(
                routing_module,
                "last_spatial_allocation_mean",
                None,
            )
            self.last_source_free_update_count = getattr(
                routing_module,
                "last_source_free_update_count",
                None,
            )
            output = {
                "mask": mask,
                "prototype_features": decoder,
            }
            if stage_weights is not None:
                output["stage_attn_res_weights"] = stage_weights
            if history_weights is not None:
                output["delta_history_weights"] = history_weights
            if self.last_expert_weights is not None:
                output["expert_weights"] = self.last_expert_weights
            if self.last_dense_expert_weights is not None:
                output["dense_expert_weights"] = self.last_dense_expert_weights
            if stage_selection_confidence is not None:
                output["stage_selection_confidence"] = stage_selection_confidence
            if delta_history_confidence is not None:
                output["delta_history_confidence"] = delta_history_confidence
            if depth_reliability is not None:
                output["depth_reliability"] = depth_reliability
            if self.last_routing_confidence is not None:
                output["routing_confidence"] = self.last_routing_confidence
            if self.last_combined_reliability is not None:
                output["combined_reliability"] = self.last_combined_reliability
            if self.last_reliability_gate is not None:
                output["reliability_gate"] = self.last_reliability_gate
            if self.last_active_expert_count is not None:
                output["active_expert_count"] = self.last_active_expert_count
            if predicted_benefits is not None:
                output["predicted_expert_benefits"] = predicted_benefits
            if counterfactual_masks is not None:
                output["counterfactual_masks"] = counterfactual_masks
            if gsd_risk is not None or self.last_gsd_risk is not None:
                output["gsd_risk"] = gsd_risk if gsd_risk is not None else self.last_gsd_risk
            if gsd_risk_logits is not None or self.last_gsd_risk_logits is not None:
                output["gsd_risk_logits"] = (
                    gsd_risk_logits
                    if gsd_risk_logits is not None
                    else self.last_gsd_risk_logits
                )
            if spatial_priority_logits is not None or self.last_spatial_priority_logits is not None:
                output["spatial_priority_logits"] = (
                    spatial_priority_logits
                    if spatial_priority_logits is not None
                    else self.last_spatial_priority_logits
                )
            if spatial_allocation is not None or self.last_spatial_allocation is not None:
                output["spatial_allocation"] = (
                    spatial_allocation
                    if spatial_allocation is not None
                    else self.last_spatial_allocation
                )
            if self.last_spatial_active_fraction is not None:
                output["spatial_active_fraction"] = self.last_spatial_active_fraction
            if self.last_spatial_allocation_mean is not None:
                output["spatial_allocation_mean"] = self.last_spatial_allocation_mean
            if self.boundary_head is not None:
                output["boundary"] = self.boundary_head(decoder)
            return output

        @torch.no_grad()
        def adapt_source_free(self, x, *, gsd=None) -> dict[str, Any]:
            """Run one unlabeled, reliability-filtered expert adaptation update."""
            output = self.forward_with_aux(x, gsd=gsd)
            if isinstance(self.spatial_moe, SharedRoutedSpatialMoE):
                self.last_source_free_update_count = self.spatial_moe.adapt_source_free_from_current()
                if self.last_source_free_update_count is not None:
                    output["source_free_update_count"] = self.last_source_free_update_count
            return output

        def forward(self, x, *, gsd=None):
            """Return mask logits for standard segmentation training and inference."""
            return self.forward_with_aux(x, gsd=gsd)["mask"]


    class AERISFeaturePyramidSegmenter(DARAFeaturePyramidSegmenter):
        """Official AERIS-Seg graph with inter-stage selection and spatial experts."""
else:  # pragma: no cover - torch missing
    class CENetLite:  # type: ignore[override]
        """Placeholder CENetLite used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch is required to construct CENetLite.")

    class TimmFeaturePyramidSegmenter:  # type: ignore[override]
        """Placeholder timm-backed segmenter used when dependencies are unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch and timm are required to construct the PVT-v2 segmenter.")


    class DARAFeaturePyramidSegmenter:  # type: ignore[override]
        """Placeholder DARA segmenter used when dependencies are unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            """Raise a dependency error on construction."""
            raise RuntimeError("Torch and timm are required to construct the DARA segmenter.")


    class AERISFeaturePyramidSegmenter(DARAFeaturePyramidSegmenter):  # type: ignore[override]
        """Placeholder AERIS segmenter used when dependencies are unavailable."""

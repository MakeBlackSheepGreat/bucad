"""Custom segmentation blocks used by paper-guided segmenter variants."""

from __future__ import annotations

from typing import Any

from src.utils.runtime import optional_import, require_dependency


torch = optional_import("torch")
nn = optional_import("torch.nn")
F = optional_import("torch.nn.functional")
timm = optional_import("timm")


if nn is not None:
    class ConvNormAct(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            *,
            kernel_size: int = 3,
            groups: int = 1,
        ) -> None:
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
            return self.block(x)


    class FeatureEdgeAmplifier(nn.Module):
        """CENet-style multi-scale difference block for skip edge enhancement."""

        def __init__(self, channels: int, *, scales: tuple[int, ...] = (2, 4)) -> None:
            super().__init__()
            self.scales = scales
            self.fuse = ConvNormAct(channels * (len(scales) + 1), channels, kernel_size=1)

        def forward(self, x):
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
            attention = torch.sigmoid(self.edge_attention(x) - self.context_attention(x))
            return x * (1.0 + attention)


    class DSEB(nn.Module):
        """Differential skip enhancement block inspired by CENet."""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.edge = FeatureEdgeAmplifier(channels)
            self.attention = DifferentialAttention(channels)
            self.output = ConvNormAct(channels, channels, kernel_size=3)

        def forward(self, x):
            enhanced = self.attention(self.edge(x))
            return self.output(x + enhanced)


    class ChannelCalibrationUnit(nn.Module):
        def __init__(self, channels: int, *, reduction: int = 8) -> None:
            super().__init__()
            hidden = max(4, channels // reduction)
            self.mlp = nn.Sequential(
                nn.Conv2d(channels * 3, hidden, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, channels, kernel_size=1),
            )

        def forward(self, x):
            avg = x.mean(dim=(2, 3), keepdim=True)
            max_values = x.amax(dim=(2, 3), keepdim=True)
            std = x.std(dim=(2, 3), keepdim=True, unbiased=False)
            weights = torch.sigmoid(self.mlp(torch.cat([avg, max_values, std], dim=1)))
            return x * (1.0 + weights)


    class MultiScaleContextAggregator(nn.Module):
        def __init__(self, channels: int, *, kernels: tuple[int, ...] = (3, 5, 7)) -> None:
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
            return self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


    class WeightedNonLocalLite(nn.Module):
        def __init__(self, channels: int, *, pool_size: int = 16) -> None:
            super().__init__()
            hidden = max(8, channels // 2)
            self.pool_size = pool_size
            self.query = nn.Conv2d(channels, hidden, kernel_size=1)
            self.key = nn.Conv2d(channels, hidden, kernel_size=1)
            self.value = nn.Conv2d(channels, channels, kernel_size=1)
            self.weight = nn.Parameter(torch.tensor(0.0))

        def forward(self, x):
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
            super().__init__()
            self.channel = ChannelCalibrationUnit(channels)
            self.context = MultiScaleContextAggregator(channels)
            self.nonlocal_block = WeightedNonLocalLite(channels) if use_nonlocal else nn.Identity()
            self.output = ConvNormAct(channels, channels, kernel_size=3)

        def forward(self, x):
            calibrated = self.channel(x)
            contextual = self.context(calibrated)
            return self.output(self.nonlocal_block(contextual) + x)


    class EncoderStage(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, *, downsample: bool) -> None:
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
            return self.block(x)


    class DecoderStage(nn.Module):
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
            super().__init__()
            self.skip_enhance = DSEB(skip_channels) if use_dseb else nn.Identity()
            self.reduce = ConvNormAct(in_channels + skip_channels, out_channels)
            self.context = CFAM(out_channels, use_nonlocal=use_nonlocal) if use_cfam else nn.Identity()
            self.output = ConvNormAct(out_channels, out_channels)

        def forward(self, x, skip):
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
            return self.forward_with_aux(x)["mask"]
else:  # pragma: no cover - torch missing
    class CENetLite:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Torch is required to construct CENetLite.")

    class TimmFeaturePyramidSegmenter:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Torch and timm are required to construct the PVT-v2 segmenter.")

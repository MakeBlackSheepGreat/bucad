"""Block Attention Residual adapters for compact ConvNeXt classifiers."""

from __future__ import annotations

import math
from typing import Any

from src.utils.runtime import optional_import, require_dependency


torch = optional_import("torch")
nn = optional_import("torch.nn")


if nn is not None:
    class RMSNorm2d(nn.Module):
        """Apply RMS normalization across channels at every spatial location."""

        def __init__(self, channels: int, *, eps: float = 1e-6) -> None:
            super().__init__()
            self.channels = int(channels)
            self.eps = float(eps)
            self.weight = nn.Parameter(torch.ones(1, self.channels, 1, 1))

        def forward(self, x):
            """Normalize one channels-first feature tensor."""
            scale = x.float().pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
            return (x * scale.to(dtype=x.dtype)) * self.weight.to(dtype=x.dtype)


    class StageBlockAttentionResidual(nn.Module):
        """Inject a depth-wise softmax history into one ConvNeXt stage."""

        def __init__(
            self,
            *,
            channels: int,
            depth: int,
            block_size: int = 2,
            injection_interval: int = 1,
            temperature: float = 1.0,
            gate_init: float = 0.0,
            query_mode: str = "pseudo",
            fusion_mode: str = "add",
            history_mode: str = "delta",
        ) -> None:
            super().__init__()
            self.channels = int(channels)
            self.depth = int(depth)
            self.block_size = max(1, int(block_size))
            self.injection_interval = max(1, int(injection_interval))
            self.temperature = max(float(temperature), 1e-4)
            self.query_mode = str(query_mode).lower()
            self.fusion_mode = str(fusion_mode).lower()
            self.history_mode = str(history_mode).lower()
            if self.query_mode not in {"pseudo", "content"}:
                raise ValueError("query_mode must be 'pseudo' or 'content'.")
            if self.fusion_mode not in {"add", "interpolate"}:
                raise ValueError("fusion_mode must be 'add' or 'interpolate'.")
            if self.history_mode not in {"delta", "state"}:
                raise ValueError("history_mode must be 'delta' or 'state'.")
            if self.channels <= 0 or self.depth <= 0:
                raise ValueError("channels and depth must be positive.")
            self.key_norm = RMSNorm2d(self.channels)
            self.pseudo_queries = nn.Parameter(torch.zeros(self.depth, self.channels))
            self.content_query = nn.Linear(self.channels, self.channels, bias=False)
            nn.init.eye_(self.content_query.weight)
            self.residual_gates = nn.Parameter(torch.full((self.depth,), float(gate_init)))
            self.last_attention_weights: list[Any] = []

        def _attention(self, values, layer_index: int):
            """Attend over stage input, completed blocks, and the current partial block."""
            stacked = torch.stack(values, dim=1)
            batch_size, history_size, channels, height, width = stacked.shape
            normalized = self.key_norm(stacked.reshape(batch_size * history_size, channels, height, width))
            normalized = normalized.reshape(batch_size, history_size, channels, height, width)
            if self.query_mode == "content":
                query_source = values[-1].mean(dim=(2, 3))
                query = self.content_query(query_source.float()).to(dtype=stacked.dtype)
                query = query / query.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
                logits = torch.einsum("bc,bnchw->bnhw", query, normalized) / math.sqrt(float(channels))
            else:
                query = self.pseudo_queries[layer_index].to(device=stacked.device, dtype=stacked.dtype)
                logits = torch.einsum("c,bnchw->bnhw", query, normalized) / math.sqrt(float(channels))
            weights = torch.softmax(logits / self.temperature, dim=1)
            context = torch.einsum("bnhw,bnchw->bchw", weights, stacked)
            return context, weights

        def forward(self, stage, x):
            """Run a ConvNeXt stage with zero-initialized history residual injection."""
            x = stage.downsample(x)
            stage_input = x
            completed_blocks = []
            partial_block = None
            self.last_attention_weights = []
            depth = len(stage.blocks)
            if depth != self.depth:
                raise RuntimeError(f"Expected stage depth {self.depth}, received {depth}.")
            for layer_index, block in enumerate(stage.blocks):
                previous = x
                block_output = block(x)
                residual_increment = block_output - previous
                partial_block = (
                    residual_increment
                    if partial_block is None
                    else partial_block + residual_increment
                )
                should_inject = (
                    (layer_index + 1) % self.injection_interval == 0
                    or layer_index + 1 == depth
                )
                if should_inject:
                    if self.history_mode == "state":
                        values = [stage_input, *completed_blocks, block_output]
                    else:
                        values = [stage_input, *completed_blocks, partial_block]
                    context, weights = self._attention(values, layer_index)
                    gate = torch.tanh(self.residual_gates[layer_index]).to(dtype=block_output.dtype)
                    if self.fusion_mode == "interpolate":
                        x = block_output + gate * (context - block_output)
                    else:
                        x = block_output + gate * context
                    self.last_attention_weights.append(weights.detach().mean(dim=(0, 2, 3)))
                else:
                    x = block_output
                if (layer_index + 1) % self.block_size == 0 or layer_index + 1 == depth:
                    completed_blocks.append(x if self.history_mode == "state" else partial_block)
                    partial_block = None
            return x


    class DepthwiseBlockAttnResConvNeXtClassifier(nn.Module):
        """Single-image ConvNeXt classifier with compact block attention residual adapters."""

        def __init__(
            self,
            *,
            backbone_name: str = "convnext_tiny",
            pretrained: bool = True,
            in_chans: int = 3,
            num_classes: int = 2,
            attnres_stages: list[int] | tuple[int, ...] = (3, 4),
            attnres_block_size: int = 2,
            attnres_inject_every: int = 1,
            attnres_temperature: float = 1.0,
            attnres_gate_init: float = 0.0,
            attnres_query_mode: str = "pseudo",
            attnres_fusion_mode: str = "add",
            attnres_history_mode: str = "delta",
            head_dropout: float = 0.2,
            **backbone_kwargs,
        ) -> None:
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
                **backbone_kwargs,
            )
            if not all(hasattr(self.backbone, attribute) for attribute in ("stem", "stages", "norm_pre")):
                raise RuntimeError(f"{self.backbone_name!r} does not expose ConvNeXt stage modules.")
            requested_stages = tuple(sorted({int(stage) for stage in attnres_stages}))
            if not requested_stages or any(stage < 1 or stage > len(self.backbone.stages) for stage in requested_stages):
                raise ValueError("attnres_stages must contain ConvNeXt stage numbers in [1, 4].")
            self.attnres_stages = requested_stages
            self.attnres = nn.ModuleDict()
            for stage_number in self.attnres_stages:
                stage = self.backbone.stages[stage_number - 1]
                blocks = getattr(stage, "blocks", None)
                if blocks is None or len(blocks) == 0:
                    raise RuntimeError(f"ConvNeXt stage {stage_number} has no residual blocks.")
                channels = self._stage_channels(stage)
                self.attnres[str(stage_number)] = StageBlockAttentionResidual(
                    channels=channels,
                    depth=len(blocks),
                    block_size=int(attnres_block_size),
                    injection_interval=int(attnres_inject_every),
                    temperature=float(attnres_temperature),
                    gate_init=float(attnres_gate_init),
                    query_mode=str(attnres_query_mode),
                    fusion_mode=str(attnres_fusion_mode),
                    history_mode=str(attnres_history_mode),
                )
            feature_dim = int(getattr(self.backbone, "num_features", 0))
            if feature_dim <= 0:
                raise RuntimeError("ConvNeXt backbone did not expose num_features.")
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head_norm = nn.LayerNorm(feature_dim)
            self.head_dropout = nn.Dropout(float(head_dropout))
            self.classifier = nn.Linear(feature_dim, int(num_classes))
            self.gradcam_layer = self.backbone.stages[-1].blocks[-1]
            self.pretrained_cfg = dict(getattr(self.backbone, "pretrained_cfg", {}) or {})
            self.default_cfg = dict(getattr(self.backbone, "default_cfg", {}) or {})

        @staticmethod
        def _stage_channels(stage) -> int:
            """Resolve the channels used by a ConvNeXt stage's first residual block."""
            block = stage.blocks[0]
            channels = int(getattr(getattr(block, "conv_dw", None), "in_channels", 0))
            if channels <= 0:
                raise RuntimeError("Unable to determine ConvNeXt stage channel count.")
            return channels

        def forward_features(self, x):
            """Run ConvNeXt while replacing selected stage residual paths with AttnRes adapters."""
            x = self.backbone.stem(x)
            for stage_number, stage in enumerate(self.backbone.stages, start=1):
                adapter = self.attnres[str(stage_number)] if str(stage_number) in self.attnres else None
                x = adapter(stage, x) if adapter is not None else stage(x)
            return self.backbone.norm_pre(x)

        def forward_with_embedding(self, x, labels=None, **_kwargs):
            """Return logits and the compact pooled embedding used by the classification head."""
            features = self.forward_features(x)
            embedding = self.pool(features).flatten(1)
            logits = self.classifier(self.head_dropout(self.head_norm(embedding)))
            return logits, embedding

        def forward(self, x, labels=None, **_kwargs):
            """Return class logits for one image batch."""
            logits, _embedding = self.forward_with_embedding(x, labels=labels)
            return logits

        def attention_summary(self) -> dict[str, list[list[float]]]:
            """Return detached spatially averaged depth attention weights for diagnostics."""
            summary: dict[str, list[list[float]]] = {}
            for stage_number, module in self.attnres.items():
                rows = []
                for weights in module.last_attention_weights:
                    rows.append([float(value) for value in weights.cpu().tolist()])
                summary[f"stage_{stage_number}"] = rows
            return summary


else:
    class DepthwiseBlockAttnResConvNeXtClassifier:  # type: ignore[override]
        """Dependency placeholder."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Torch is required to construct DepthwiseBlockAttnResConvNeXtClassifier.")

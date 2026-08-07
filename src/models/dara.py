"""AERIS mechanisms with historical DARA/QDHE parameter compatibility."""

from __future__ import annotations

import math
from typing import Any

from src.utils.runtime import optional_import


torch = optional_import("torch")
nn = optional_import("torch.nn")
F = optional_import("torch.nn.functional")


if nn is not None:
    class StageAttentionResidualFusion(nn.Module):
        """Apply content-dependent residual weights across stage embeddings."""

        def __init__(
            self,
            stage_dims: list[int] | tuple[int, ...],
            *,
            attention_dim: int = 128,
            temperature: float = 1.0,
            residual_strength: float = 1.0,
            query_mode: str = "content",
            descriptor_dim: int = 0,
            initial_weights: list[float] | tuple[float, ...] | None = None,
            dropout: float = 0.0,
        ) -> None:
            """Build a Block-AttnRes-inspired selector over CNN stage tokens."""
            super().__init__()
            self.stage_dims = tuple(int(dim) for dim in stage_dims)
            if len(self.stage_dims) == 0 or any(dim <= 0 for dim in self.stage_dims):
                raise ValueError("stage_dims must contain positive dimensions.")
            self.num_stages = len(self.stage_dims)
            self.attention_dim = int(attention_dim)
            if self.attention_dim <= 0:
                raise ValueError("attention_dim must be positive.")
            resolved_mode = str(query_mode).lower()
            if resolved_mode not in {"learned", "content", "descriptor", "hybrid"}:
                raise ValueError("query_mode must be learned, content, descriptor, or hybrid.")
            self.query_mode = resolved_mode
            self.descriptor_dim = max(0, int(descriptor_dim))
            self.temperature = max(float(temperature), 1e-4)
            self.residual_strength = max(0.0, float(residual_strength))
            self.stage_projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, self.attention_dim, bias=False),
                    )
                    for dim in self.stage_dims
                ]
            )
            self.learned_query = nn.Parameter(torch.zeros(self.attention_dim))
            nn.init.normal_(self.learned_query, std=0.02)
            self.descriptor_query = (
                nn.Sequential(
                    nn.LayerNorm(self.descriptor_dim),
                    nn.Linear(self.descriptor_dim, self.attention_dim),
                    nn.GELU(),
                    nn.Linear(self.attention_dim, self.attention_dim),
                )
                if self.descriptor_dim > 0
                else None
            )
            self.dropout = nn.Dropout(float(dropout))
            if initial_weights is None:
                initial = torch.ones(self.num_stages, dtype=torch.float32)
            else:
                if len(initial_weights) != self.num_stages:
                    raise ValueError("initial_weights must match stage_dims.")
                initial = torch.tensor(initial_weights, dtype=torch.float32).clamp_min(1e-4)
            self.stage_prior_logits = nn.Parameter(initial.log())
            self.last_attention_weights = None
            self.last_residual_scales = None

        def _normalize_descriptor(self, descriptor, reference):
            """Align an optional descriptor tensor to the stage-token batch."""
            if self.descriptor_dim <= 0:
                return None
            if descriptor is None:
                return reference.new_zeros((reference.shape[0], self.descriptor_dim))
            descriptor = descriptor.to(device=reference.device, dtype=reference.dtype)
            if descriptor.ndim == 1:
                descriptor = descriptor.unsqueeze(0)
            if descriptor.shape[0] == 1 and reference.shape[0] > 1:
                descriptor = descriptor.expand(reference.shape[0], -1)
            if descriptor.shape != (reference.shape[0], self.descriptor_dim):
                raise ValueError(
                    f"Expected descriptor shape [B,{self.descriptor_dim}], got {tuple(descriptor.shape)}."
                )
            return descriptor

        def _query(self, projected_tokens, descriptor):
            """Construct the learned, content, descriptor, or hybrid query."""
            batch_size = projected_tokens.shape[0]
            query = self.learned_query.to(
                device=projected_tokens.device,
                dtype=projected_tokens.dtype,
            ).unsqueeze(0).expand(batch_size, -1)
            content_query = projected_tokens.mean(dim=1)
            if self.query_mode in {"content", "hybrid"}:
                query = query + content_query
            if self.query_mode in {"descriptor", "hybrid"}:
                normalized_descriptor = self._normalize_descriptor(descriptor, projected_tokens)
                if normalized_descriptor is None or self.descriptor_query is None:
                    query = query + content_query
                else:
                    query = query + self.descriptor_query(normalized_descriptor)
            return F.normalize(query, dim=-1)

        def forward(self, embeddings: list[Any], descriptor=None) -> tuple[list[Any], Any]:
            """Return residual-scaled stage embeddings and per-sample weights."""
            if len(embeddings) != self.num_stages:
                raise ValueError(
                    f"Expected {self.num_stages} stage embeddings, got {len(embeddings)}."
                )
            projected = []
            for index, (embedding, projection) in enumerate(zip(embeddings, self.stage_projections)):
                if embedding.ndim != 2 or embedding.shape[1] != self.stage_dims[index]:
                    raise ValueError(
                        f"Stage {index} expected [B,{self.stage_dims[index]}], got {tuple(embedding.shape)}."
                    )
                projected.append(F.normalize(projection(embedding), dim=-1))
            projected_tokens = torch.stack(projected, dim=1)
            query = self._query(projected_tokens, descriptor)
            logits = torch.einsum("bsd,bd->bs", projected_tokens, query) / math.sqrt(self.attention_dim)
            logits = logits + self.stage_prior_logits.to(device=logits.device, dtype=logits.dtype)
            weights = torch.softmax(logits / self.temperature, dim=1) * float(self.num_stages)
            weights = self.dropout(weights)
            residual_scales = 1.0 + self.residual_strength * (weights - 1.0)
            scaled = [
                embedding * residual_scales[:, index : index + 1]
                for index, embedding in enumerate(embeddings)
            ]
            self.last_attention_weights = weights.detach()
            self.last_residual_scales = residual_scales.detach()
            return scaled, weights


    class CrossStageDeltaHistoryFusion(nn.Module):
        """Inject selectively routed stage deltas into later stage embeddings."""

        def __init__(
            self,
            stage_dims: list[int] | tuple[int, ...],
            *,
            history_dim: int = 128,
            temperature: float = 1.0,
            residual_strength: float = 0.5,
            descriptor_dim: int = 0,
        ) -> None:
            """Build causal cross-stage routing over incremental representations."""
            super().__init__()
            self.stage_dims = tuple(int(dim) for dim in stage_dims)
            if len(self.stage_dims) < 2 or any(dim <= 0 for dim in self.stage_dims):
                raise ValueError("stage_dims must contain at least two positive dimensions.")
            self.num_stages = len(self.stage_dims)
            self.history_dim = int(history_dim)
            if self.history_dim <= 0:
                raise ValueError("history_dim must be positive.")
            self.temperature = max(float(temperature), 1e-4)
            self.residual_strength = max(0.0, float(residual_strength))
            self.descriptor_dim = max(0, int(descriptor_dim))
            self.stage_projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, self.history_dim, bias=False),
                    )
                    for dim in self.stage_dims
                ]
            )
            self.history_keys = nn.Sequential(
                nn.LayerNorm(self.history_dim),
                nn.Linear(self.history_dim, self.history_dim, bias=False),
            )
            self.stage_queries = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(self.history_dim),
                        nn.Linear(self.history_dim, self.history_dim, bias=False),
                    )
                    for _ in self.stage_dims
                ]
            )
            self.history_outputs = nn.ModuleList(
                [nn.Linear(self.history_dim, dim, bias=False) for dim in self.stage_dims]
            )
            self.learned_queries = nn.Parameter(torch.zeros(self.num_stages, self.history_dim))
            nn.init.normal_(self.learned_queries, std=0.02)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(5.0)))
            self.residual_gates = nn.Parameter(torch.full((self.num_stages,), 0.1))
            self.descriptor_query = (
                nn.Sequential(
                    nn.LayerNorm(self.descriptor_dim),
                    nn.Linear(self.descriptor_dim, self.history_dim),
                    nn.GELU(),
                    nn.Linear(self.history_dim, self.history_dim),
                )
                if self.descriptor_dim > 0
                else None
            )
            self.last_history_weights = None
            self.last_delta_tokens = None

        def _descriptor_embedding(self, descriptor, reference):
            """Return one optional descriptor query per sample."""
            if self.descriptor_query is None or self.descriptor_dim <= 0:
                return None
            if descriptor is None:
                descriptor = reference.new_zeros((reference.shape[0], self.descriptor_dim))
            else:
                descriptor = descriptor.to(device=reference.device, dtype=reference.dtype)
                if descriptor.ndim == 1:
                    descriptor = descriptor.unsqueeze(0)
                if descriptor.shape[0] == 1 and reference.shape[0] > 1:
                    descriptor = descriptor.expand(reference.shape[0], -1)
                if descriptor.shape != (reference.shape[0], self.descriptor_dim):
                    raise ValueError(
                        f"Expected descriptor shape [B,{self.descriptor_dim}], got {tuple(descriptor.shape)}."
                    )
            return self.descriptor_query(descriptor)

        def forward(self, embeddings: list[Any], descriptor=None) -> tuple[list[Any], Any]:
            """Return delta-history-enhanced embeddings and causal routing weights."""
            if len(embeddings) != self.num_stages:
                raise ValueError(f"Expected {self.num_stages} embeddings, got {len(embeddings)}.")
            tokens = []
            for index, (embedding, projection) in enumerate(zip(embeddings, self.stage_projections)):
                if embedding.ndim != 2 or embedding.shape[1] != self.stage_dims[index]:
                    raise ValueError(
                        f"Stage {index} expected [B,{self.stage_dims[index]}], got {tuple(embedding.shape)}."
                    )
                tokens.append(projection(embedding))
            token_stack = torch.stack(tokens, dim=1)
            delta_stack = torch.cat(
                [token_stack[:, :1], token_stack[:, 1:] - token_stack[:, :-1]],
                dim=1,
            )
            keys = F.normalize(self.history_keys(delta_stack), dim=-1)
            descriptor_query = self._descriptor_embedding(descriptor, token_stack)
            query_rows = []
            for index, query_layer in enumerate(self.stage_queries):
                query = query_layer(token_stack[:, index]) + self.learned_queries[index]
                if descriptor_query is not None:
                    query = query + descriptor_query
                query_rows.append(F.normalize(query, dim=-1))
            queries = torch.stack(query_rows, dim=1)
            logit_scale = self.logit_scale.exp().clamp(max=100.0)
            scores = torch.einsum("bid,bjd->bij", queries, keys) * logit_scale
            causal_mask = torch.triu(
                torch.ones(
                    self.num_stages,
                    self.num_stages,
                    device=scores.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0), torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores / self.temperature, dim=-1)
            contexts = torch.bmm(weights, delta_stack)
            enhanced = []
            for index, (embedding, output_layer) in enumerate(zip(embeddings, self.history_outputs)):
                gate = torch.tanh(self.residual_gates[index]) * self.residual_strength
                enhanced.append(embedding + gate * output_layer(contexts[:, index]))
            self.last_history_weights = weights.detach()
            self.last_delta_tokens = delta_stack.detach()
            return enhanced, weights


    class CrossStageDeltaHistory2d(nn.Module):
        """Route aligned spatial feature deltas through a causal stage history."""

        def __init__(
            self,
            channels: int,
            num_stages: int,
            *,
            attention_dim: int = 64,
            temperature: float = 1.0,
            residual_strength: float = 0.5,
        ) -> None:
            """Build spatial delta-history injection for feature pyramids."""
            super().__init__()
            self.channels = int(channels)
            self.num_stages = int(num_stages)
            if self.channels <= 0 or self.num_stages < 2:
                raise ValueError("channels must be positive and num_stages must be at least two.")
            self.attention_dim = int(attention_dim)
            self.temperature = max(float(temperature), 1e-4)
            self.residual_strength = max(0.0, float(residual_strength))
            self.query = nn.Linear(self.channels, self.attention_dim, bias=False)
            self.key = nn.Linear(self.channels, self.attention_dim, bias=False)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(5.0)))
            self.residual_gates = nn.Parameter(torch.full((self.num_stages,), 0.1))
            self.last_history_weights = None

        def forward(self, features: list[Any]) -> tuple[list[Any], Any]:
            """Return spatial features enhanced by aligned incremental history."""
            if len(features) != self.num_stages:
                raise ValueError(f"Expected {self.num_stages} features, got {len(features)}.")
            batch_size = features[0].shape[0]
            weight_rows = features[0].new_zeros((batch_size, self.num_stages, self.num_stages))
            enhanced = [features[0]]
            weight_rows[:, 0, 0] = 1.0
            for target_index in range(1, self.num_stages):
                target = features[target_index]
                aligned = [
                    feature
                    if feature.shape[-2:] == target.shape[-2:]
                    else F.interpolate(feature, size=target.shape[-2:], mode="bilinear", align_corners=False)
                    for feature in features[: target_index + 1]
                ]
                delta_maps = [aligned[0]]
                delta_maps.extend(aligned[index] - aligned[index - 1] for index in range(1, len(aligned)))
                delta_stack = torch.stack(delta_maps, dim=1)
                query = F.normalize(
                    self.query(F.adaptive_avg_pool2d(target, 1).flatten(1)),
                    dim=-1,
                )
                key_tokens = torch.stack(
                    [F.adaptive_avg_pool2d(delta, 1).flatten(1) for delta in delta_maps],
                    dim=1,
                )
                keys = F.normalize(self.key(key_tokens), dim=-1)
                logit_scale = self.logit_scale.exp().clamp(max=100.0)
                scores = torch.einsum("bd,bjd->bj", query, keys) * logit_scale
                weights = torch.softmax(scores / self.temperature, dim=1)
                context = (delta_stack * weights[:, :, None, None, None]).sum(dim=1)
                gate = torch.tanh(self.residual_gates[target_index]) * self.residual_strength
                enhanced.append(target + gate * context)
                weight_rows[:, target_index, : target_index + 1] = weights
            self.last_history_weights = weight_rows.detach()
            return enhanced, weight_rows


    class SharedRoutedExpertHead(nn.Module):
        """Combine one shared classifier with fine-grained routed experts."""

        def __init__(
            self,
            input_dim: int,
            num_classes: int,
            *,
            shared_hidden_dim: int = 256,
            expert_hidden_dim: int = 128,
            router_hidden_dim: int = 128,
            num_routed_experts: int = 4,
            top_k: int = 2,
            router_temperature: float = 1.0,
            routed_scale: float = 0.5,
            dropout: float = 0.1,
        ) -> None:
            """Build stable shared and sparse-routed prediction paths."""
            super().__init__()
            self.input_dim = int(input_dim)
            self.num_classes = int(num_classes)
            self.num_routed_experts = max(1, int(num_routed_experts))
            self.top_k = min(max(1, int(top_k)), self.num_routed_experts)
            self.router_temperature = max(float(router_temperature), 1e-4)
            self.routed_scale = float(routed_scale)

            def _expert(hidden_dim: int, *, zero_output: bool = False):
                output = nn.Linear(int(hidden_dim), self.num_classes)
                if zero_output:
                    nn.init.zeros_(output.weight)
                    nn.init.zeros_(output.bias)
                return nn.Sequential(
                    nn.LayerNorm(self.input_dim),
                    nn.Linear(self.input_dim, int(hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    output,
                )

            self.shared_expert = _expert(shared_hidden_dim)
            self.routed_experts = nn.ModuleList(
                [
                    _expert(expert_hidden_dim, zero_output=True)
                    for _ in range(self.num_routed_experts)
                ]
            )
            self.router = nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, int(router_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(router_hidden_dim), self.num_routed_experts),
            )
            nn.init.normal_(self.router[-1].weight, std=1e-3)
            nn.init.zeros_(self.router[-1].bias)
            self.last_dense_expert_weights = None
            self.last_dense_expert_weights_detached = None
            self.last_expert_weights = None
            self.last_expert_weights_detached = None
            self.last_shared_logits = None
            self.last_routed_logits = None
            self.last_active_expert_count = None

        def _routing_weights(self, embedding):
            """Return normalized top-k routing weights."""
            router_logits = self.router(embedding) / self.router_temperature
            weights = torch.softmax(router_logits, dim=1)
            self.last_dense_expert_weights = weights
            self.last_dense_expert_weights_detached = weights.detach()
            if self.top_k < self.num_routed_experts:
                top_values, top_indices = torch.topk(weights, k=self.top_k, dim=1)
                sparse = torch.zeros_like(weights).scatter(1, top_indices, top_values)
                weights = sparse / sparse.sum(dim=1, keepdim=True).clamp_min(1e-8)
            return weights

        def forward(self, embedding):
            """Return shared logits plus the weighted routed-expert residual."""
            shared_logits = self.shared_expert(embedding)
            expert_weights = self._routing_weights(embedding)
            routed_logits = embedding.new_zeros((embedding.shape[0], self.num_classes))
            for expert_index, expert in enumerate(self.routed_experts):
                selected = expert_weights[:, expert_index] > 0
                if selected.any():
                    routed_logits[selected] += (
                        expert(embedding[selected])
                        * expert_weights[selected, expert_index].unsqueeze(1)
                    )
            logits = shared_logits + self.routed_scale * routed_logits
            self.last_expert_weights = expert_weights
            self.last_expert_weights_detached = expert_weights.detach()
            self.last_shared_logits = shared_logits.detach()
            self.last_routed_logits = routed_logits.detach()
            self.last_active_expert_count = (expert_weights > 0).sum(dim=1).detach()
            return logits


    class ReliabilityGatedSharedRoutedExpertHead(SharedRoutedExpertHead):
        """Suppress routed experts when routing evidence is uncertain."""

        def __init__(
            self,
            *args,
            reliability_threshold: float = 0.15,
            reliability_temperature: float = 0.1,
            use_reliability_gate: bool = True,
            **kwargs,
        ) -> None:
            """Build a shared-routed head with continuous quality fallback."""
            super().__init__(*args, **kwargs)
            self.reliability_threshold = float(reliability_threshold)
            self.reliability_temperature = max(float(reliability_temperature), 1e-4)
            self.use_reliability_gate = bool(use_reliability_gate)
            self.last_routing_confidence = None
            self.last_combined_reliability = None
            self.last_reliability_gate = None

        def forward(self, embedding, external_reliability=None):
            """Return logits with confidence-controlled routed contribution."""
            shared_logits = self.shared_expert(embedding)
            dense_weights = torch.softmax(self.router(embedding) / self.router_temperature, dim=1)
            self.last_dense_expert_weights = dense_weights
            self.last_dense_expert_weights_detached = dense_weights.detach()
            entropy = -(dense_weights.clamp_min(1e-8) * dense_weights.clamp_min(1e-8).log()).sum(dim=1)
            max_entropy = math.log(float(self.num_routed_experts))
            routing_confidence = 1.0 - entropy / max(max_entropy, 1e-8)
            reliability = routing_confidence
            if external_reliability is not None:
                external_reliability = external_reliability.to(
                    device=embedding.device,
                    dtype=embedding.dtype,
                ).reshape(-1)
                reliability = 0.5 * (reliability + external_reliability.clamp(0.0, 1.0))
            gate = (
                torch.sigmoid(
                    (reliability - self.reliability_threshold) / self.reliability_temperature
                )
                if self.use_reliability_gate
                else torch.ones_like(reliability)
            )
            expert_weights = dense_weights
            if self.top_k < self.num_routed_experts:
                top_values, top_indices = torch.topk(expert_weights, k=self.top_k, dim=1)
                sparse = torch.zeros_like(expert_weights).scatter(1, top_indices, top_values)
                expert_weights = sparse / sparse.sum(dim=1, keepdim=True).clamp_min(1e-8)
            routed_logits = embedding.new_zeros((embedding.shape[0], self.num_classes))
            for expert_index, expert in enumerate(self.routed_experts):
                selected = expert_weights[:, expert_index] > 0
                if selected.any():
                    routed_logits[selected] += (
                        expert(embedding[selected])
                        * expert_weights[selected, expert_index].unsqueeze(1)
                    )
            logits = shared_logits + gate.unsqueeze(1) * self.routed_scale * routed_logits
            self.last_expert_weights = expert_weights
            self.last_expert_weights_detached = expert_weights.detach()
            self.last_shared_logits = shared_logits.detach()
            self.last_routed_logits = routed_logits.detach()
            self.last_routing_confidence = routing_confidence.detach()
            self.last_combined_reliability = reliability.detach()
            self.last_reliability_gate = gate.detach()
            self.last_active_expert_count = (expert_weights > 0).sum(dim=1).detach()
            return logits


    class SharedRoutedSpatialMoE(nn.Module):
        """Refine a feature map with shared and routed dilation experts."""

        def __init__(
            self,
            channels: int,
            *,
            num_routed_experts: int = 4,
            top_k: int = 2,
            dilations: list[int] | tuple[int, ...] = (1, 2, 3, 5),
            router_temperature: float = 1.0,
            routed_scale: float = 0.5,
            shared_residual_scale: float = 0.1,
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
            """Build a lightweight spatial MoE with per-image routing."""
            super().__init__()
            self.channels = int(channels)
            self.num_routed_experts = max(1, int(num_routed_experts))
            self.top_k = min(max(1, int(top_k)), self.num_routed_experts)
            self.router_temperature = max(float(router_temperature), 1e-4)
            self.routed_scale = float(routed_scale)
            initial_shared_scale = min(max(float(shared_residual_scale), 1e-4), 1.0 - 1e-4)
            self.shared_scale_logit = nn.Parameter(
                torch.tensor(math.log(initial_shared_scale / (1.0 - initial_shared_scale)))
            )
            self.use_reliability_gate = bool(use_reliability_gate)
            self.reliability_threshold = float(reliability_threshold)
            self.reliability_temperature = max(float(reliability_temperature), 1e-4)
            self.adaptive_compute = bool(adaptive_compute)
            self.min_active_experts = min(
                max(0, int(min_active_experts)),
                self.top_k,
            )
            self.gsd_adaptive_compute = bool(gsd_adaptive_compute)
            self.gsd_reference_meters = max(float(gsd_reference_meters), 1e-6)
            self.gsd_compute_gain = max(0.0, float(gsd_compute_gain))
            self.source_free_adaptation = bool(source_free_adaptation)
            self.source_free_momentum = min(max(float(source_free_momentum), 0.0), 0.9999)
            self.source_free_min_reliability = min(
                max(float(source_free_min_reliability), 0.0),
                1.0,
            )
            self.source_free_strength = max(0.0, float(source_free_strength))
            self.spatial_allocation = bool(spatial_allocation)
            self.spatial_allocation_strength = max(0.0, float(spatial_allocation_strength))
            resolved_dilations = [max(1, int(value)) for value in dilations]
            if len(resolved_dilations) == 0:
                resolved_dilations = [1]
            while len(resolved_dilations) < self.num_routed_experts:
                resolved_dilations.append(resolved_dilations[-1])

            def _block(dilation: int):
                return nn.Sequential(
                    nn.Conv2d(
                        self.channels,
                        self.channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=self.channels,
                        bias=False,
                    ),
                    nn.GroupNorm(1, self.channels),
                    nn.GELU(),
                    nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False),
                    nn.GroupNorm(1, self.channels),
                    nn.GELU(),
                )

            self.shared_expert = _block(1)
            self.routed_experts = nn.ModuleList(
                [_block(resolved_dilations[index]) for index in range(self.num_routed_experts)]
            )
            router_hidden = max(16, self.channels // 2)
            self.router = nn.Sequential(
                nn.LayerNorm(self.channels),
                nn.Linear(self.channels, router_hidden),
                nn.GELU(),
                nn.Linear(router_hidden, self.num_routed_experts),
            )
            nn.init.normal_(self.router[-1].weight, std=1e-3)
            nn.init.zeros_(self.router[-1].bias)
            resolved_thresholds = [float(value) for value in benefit_thresholds]
            if not resolved_thresholds:
                resolved_thresholds = [0.0]
            while len(resolved_thresholds) < self.top_k:
                resolved_thresholds.append(resolved_thresholds[-1])
            self.register_buffer(
                "benefit_thresholds",
                torch.tensor(resolved_thresholds[: self.top_k], dtype=torch.float32),
                persistent=False,
            )
            benefit_hidden = max(8, int(benefit_hidden_dim))
            gsd_feature_dim = 1 if self.gsd_adaptive_compute else 0
            self.benefit_head = (
                nn.Sequential(
                    nn.LayerNorm(self.channels + 2 + gsd_feature_dim),
                    nn.Linear(self.channels + 2 + gsd_feature_dim, benefit_hidden),
                    nn.GELU(),
                    nn.Linear(benefit_hidden, self.top_k),
                )
                if self.adaptive_compute
                else None
            )
            if self.benefit_head is not None:
                nn.init.normal_(self.benefit_head[-1].weight, std=1e-3)
                nn.init.zeros_(self.benefit_head[-1].bias)
            self.gsd_risk_head = (
                nn.Sequential(
                    nn.LayerNorm(self.channels + 2),
                    nn.Linear(self.channels + 2, benefit_hidden),
                    nn.GELU(),
                    nn.Linear(benefit_hidden, 1),
                )
                if self.gsd_adaptive_compute
                else None
            )
            if self.gsd_risk_head is not None:
                nn.init.normal_(self.gsd_risk_head[-1].weight, std=1e-3)
                nn.init.zeros_(self.gsd_risk_head[-1].bias)
            if self.source_free_adaptation:
                self.expert_adapter_log_scales = nn.Parameter(
                    torch.zeros(self.num_routed_experts, self.channels)
                )
                self.expert_adapter_biases = nn.Parameter(
                    torch.zeros(self.num_routed_experts, self.channels)
                )
                self.register_buffer(
                    "source_feature_anchors",
                    torch.zeros(self.num_routed_experts, self.channels),
                )
                self.register_buffer(
                    "source_anchor_ready",
                    torch.zeros(self.num_routed_experts, dtype=torch.bool),
                )
                self.register_buffer(
                    "target_feature_offsets",
                    torch.zeros(self.num_routed_experts, self.channels),
                )
                self.register_buffer(
                    "source_free_update_steps",
                    torch.zeros(self.num_routed_experts, dtype=torch.long),
                )
            else:
                self.expert_adapter_log_scales = None
                self.expert_adapter_biases = None
            self.spatial_allocation_head = (
                nn.Sequential(
                    nn.Conv2d(self.channels, max(8, self.channels // 4), kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(max(8, self.channels // 4), self.num_routed_experts + 2, kernel_size=1),
                )
                if self.spatial_allocation
                else None
            )
            if self.spatial_allocation_head is not None:
                nn.init.zeros_(self.spatial_allocation_head[-1].weight)
                nn.init.zeros_(self.spatial_allocation_head[-1].bias)
            self.last_expert_weights = None
            self.last_expert_weights_detached = None
            self.last_dense_expert_weights = None
            self.last_dense_expert_weights_detached = None
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
            self._cached_source_free_descriptor = None
            self._cached_source_free_weights = None
            self._cached_source_free_reliability = None

        def _aggregate_experts(self, feature, weights, expert_outputs, spatial_factor=None):
            """Aggregate cached expert features with one sparse weight matrix."""
            routed = torch.zeros_like(feature)
            for expert_index, expert_output in expert_outputs.items():
                selected = weights[:, expert_index] > 0
                if selected.any():
                    expert_weights = weights[selected, expert_index]
                    if spatial_factor is not None:
                        expert_weights = expert_weights[:, None, None] * spatial_factor[
                            selected,
                            expert_index,
                        ]
                    else:
                        expert_weights = expert_weights[:, None, None]
                    contribution = (
                        expert_output[selected]
                        * expert_weights[:, None]
                    ).to(dtype=routed.dtype)
                    routed[selected] = routed[selected] + contribution
            return routed

        def _normalized_gsd(self, gsd, feature):
            """Return signed log-GSD distance and a validated per-image GSD vector."""
            if gsd is None:
                values = feature.new_full((feature.shape[0],), self.gsd_reference_meters)
            else:
                values = gsd.to(device=feature.device, dtype=feature.dtype).reshape(-1)
                if values.numel() == 1 and feature.shape[0] > 1:
                    values = values.expand(feature.shape[0])
                if values.shape[0] != feature.shape[0]:
                    raise ValueError("gsd must provide one positive value per image.")
                if bool((values <= 0).any()):
                    raise ValueError("gsd values must be positive.")
            log_ratio = torch.log(values / self.gsd_reference_meters)
            return log_ratio, values

        @torch.no_grad()
        def _update_source_anchors(self, descriptor, weights) -> None:
            """Track source-domain expert descriptors while supervised training is active."""
            if not self.source_free_adaptation:
                return
            for expert_index in range(self.num_routed_experts):
                expert_weights = weights[:, expert_index]
                denominator = expert_weights.sum()
                if float(denominator.item()) <= 0.0:
                    continue
                mean_descriptor = (
                    expert_weights[:, None] * descriptor
                ).sum(dim=0) / denominator.clamp_min(1e-8)
                if bool(self.source_anchor_ready[expert_index]):
                    self.source_feature_anchors[expert_index].mul_(self.source_free_momentum).add_(
                        mean_descriptor * (1.0 - self.source_free_momentum)
                    )
                else:
                    self.source_feature_anchors[expert_index].copy_(mean_descriptor)
                    self.source_anchor_ready[expert_index] = True

        @torch.no_grad()
        def adapt_source_free_from_current(self):
            """Update selected expert offsets from reliable unlabeled target features only."""
            if (
                not self.source_free_adaptation
                or self._cached_source_free_descriptor is None
                or self._cached_source_free_weights is None
                or self._cached_source_free_reliability is None
            ):
                return None
            descriptor = self._cached_source_free_descriptor
            weights = self._cached_source_free_weights
            reliability = self._cached_source_free_reliability
            trusted = (reliability >= self.source_free_min_reliability).to(dtype=weights.dtype)
            update_counts = torch.zeros(
                self.num_routed_experts,
                device=descriptor.device,
                dtype=torch.long,
            )
            for expert_index in range(self.num_routed_experts):
                if not bool(self.source_anchor_ready[expert_index]):
                    continue
                adaptation_weights = weights[:, expert_index] * trusted
                denominator = adaptation_weights.sum()
                if float(denominator.item()) <= 0.0:
                    continue
                target_descriptor = (
                    adaptation_weights[:, None] * descriptor
                ).sum(dim=0) / denominator.clamp_min(1e-8)
                offset = target_descriptor - self.source_feature_anchors[expert_index]
                self.target_feature_offsets[expert_index].mul_(self.source_free_momentum).add_(
                    offset * (1.0 - self.source_free_momentum)
                )
                self.source_free_update_steps[expert_index].add_(1)
                update_counts[expert_index] = 1
            self.last_source_free_update_count = update_counts.sum().detach()
            return self.last_source_free_update_count

        def _expert_input(self, feature, expert_index, selected):
            """Apply expert-specific affine and target-domain offsets before one expert call."""
            inputs = feature[selected]
            if not self.source_free_adaptation:
                return inputs
            scale = torch.tanh(self.expert_adapter_log_scales[expert_index])[None, :, None, None]
            bias = self.expert_adapter_biases[expert_index][None, :, None, None]
            inputs = inputs * (1.0 + scale) + bias
            if not self.training and bool(self.source_anchor_ready[expert_index]):
                inputs = inputs + self.source_free_strength * self.target_feature_offsets[
                    expert_index
                ][None, :, None, None]
            return inputs

        def _edge_priority_prior(self, feature):
            """Produce a normalized local-contrast prior for boundary-sensitive allocation."""
            summary = feature.mean(dim=1, keepdim=True)
            vertical = torch.zeros_like(summary)
            horizontal = torch.zeros_like(summary)
            vertical[:, :, 1:] = (summary[:, :, 1:] - summary[:, :, :-1]).abs()
            horizontal[:, :, :, 1:] = (summary[:, :, :, 1:] - summary[:, :, :, :-1]).abs()
            magnitude = vertical + horizontal
            denominator = magnitude.amax(dim=(2, 3), keepdim=True).clamp_min(1e-8)
            return magnitude / denominator

        def _weights_for_active_counts(self, top_values, top_indices, active_counts):
            """Return normalized expert weights for per-sample active counts."""
            ranks = torch.arange(self.top_k, device=top_values.device).unsqueeze(0)
            rank_mask = ranks < active_counts[:, None]
            selected_values = top_values * rank_mask.to(dtype=top_values.dtype)
            sparse = torch.zeros(
                (top_values.shape[0], self.num_routed_experts),
                device=top_values.device,
                dtype=top_values.dtype,
            ).scatter(1, top_indices, selected_values)
            denominator = sparse.sum(dim=1, keepdim=True)
            return torch.where(
                denominator > 0,
                sparse / denominator.clamp_min(1e-8),
                sparse,
            )

        def forward(
            self,
            feature,
            external_reliability=None,
            *,
            gsd=None,
            return_counterfactuals: bool = False,
            force_active_experts: int | None = None,
        ):
            """Return routed features and optional counterfactual compute branches."""
            descriptor = F.adaptive_avg_pool2d(feature, 1).flatten(1)
            dense_weights = torch.softmax(self.router(descriptor) / self.router_temperature, dim=1)
            self.last_dense_expert_weights = dense_weights
            self.last_dense_expert_weights_detached = dense_weights.detach()
            entropy = -(dense_weights.clamp_min(1e-8) * dense_weights.clamp_min(1e-8).log()).sum(dim=1)
            max_entropy = math.log(float(self.num_routed_experts))
            confidence = 1.0 - entropy / max(max_entropy, 1e-8)
            reliability = confidence
            normalized_external = torch.zeros_like(confidence)
            if external_reliability is not None:
                normalized_external = external_reliability.to(
                    device=feature.device,
                    dtype=feature.dtype,
                ).reshape(-1)
                normalized_external = normalized_external.clamp(0.0, 1.0)
                reliability = 0.5 * (reliability + normalized_external)
            gate = (
                torch.sigmoid(
                    (reliability - self.reliability_threshold) / self.reliability_temperature
                )
                if self.use_reliability_gate
                else torch.ones_like(confidence)
            )
            log_gsd, _gsd_values = self._normalized_gsd(gsd, feature)
            gsd_risk_logits = (
                self.gsd_risk_head(
                        torch.cat(
                            [descriptor, log_gsd[:, None], log_gsd.abs()[:, None]],
                            dim=1,
                        )
                    ).squeeze(1)
                if self.gsd_risk_head is not None
                else None
            )
            gsd_risk = (
                0.5
                * (
                    torch.sigmoid(gsd_risk_logits)
                    + (1.0 - torch.exp(-log_gsd.abs()))
                )
                if gsd_risk_logits is not None
                else None
            )
            benefit_inputs = [descriptor, confidence[:, None], normalized_external[:, None]]
            if gsd_risk is not None:
                benefit_inputs.append(gsd_risk[:, None])
            predicted_benefits = (
                self.benefit_head(torch.cat(benefit_inputs, dim=1))
                if self.benefit_head is not None
                else None
            )
            top_values, top_indices = torch.topk(
                dense_weights,
                k=self.top_k,
                dim=1,
            )
            if force_active_experts is not None:
                active_counts = torch.full(
                    (feature.shape[0],),
                    min(max(0, int(force_active_experts)), self.top_k),
                    device=feature.device,
                    dtype=torch.long,
                )
            elif self.adaptive_compute and not self.training and predicted_benefits is not None:
                thresholds = self.benefit_thresholds.to(
                    device=predicted_benefits.device,
                    dtype=predicted_benefits.dtype,
                )
                compute_scores = predicted_benefits
                if gsd_risk is not None:
                    compute_scores = compute_scores + self.gsd_compute_gain * gsd_risk[:, None]
                incremental = compute_scores > thresholds[None, :]
                incremental = torch.cumprod(incremental.to(torch.long), dim=1)
                active_counts = torch.maximum(
                    incremental.sum(dim=1),
                    torch.full_like(incremental[:, 0], self.min_active_experts),
                )
            else:
                active_counts = torch.full(
                    (feature.shape[0],),
                    self.top_k,
                    device=feature.device,
                    dtype=torch.long,
                )
            weights = self._weights_for_active_counts(
                top_values,
                top_indices,
                active_counts,
            )
            needed = weights > 0
            if return_counterfactuals:
                needed = torch.zeros_like(needed).scatter(
                    1,
                    top_indices,
                    torch.ones_like(top_values, dtype=torch.bool),
                )
            spatial_priority_logits = None
            spatial_factor = None
            spatial_activity = None
            if self.spatial_allocation_head is not None:
                spatial_logits = self.spatial_allocation_head(feature)
                spatial_priority_logits = spatial_logits[:, :2]
                boundary_priority = torch.sigmoid(spatial_priority_logits[:, :1])
                rare_priority = torch.sigmoid(spatial_priority_logits[:, 1:2])
                learned_priority = 0.5 * (boundary_priority + rare_priority)
                spatial_priority = 0.5 * (
                    learned_priority + self._edge_priority_prior(feature)
                )
                expert_priority = torch.sigmoid(spatial_logits[:, 2:])
                spatial_activity = spatial_priority * expert_priority
                spatial_center = spatial_activity.mean(dim=(2, 3), keepdim=True)
                spatial_factor = 1.0 + self.spatial_allocation_strength * (
                    spatial_activity - spatial_center
                )
            expert_outputs = {}
            for expert_index, expert in enumerate(self.routed_experts):
                selected = needed[:, expert_index]
                if selected.any():
                    expert_output = torch.zeros_like(feature)
                    expert_output[selected] = expert(
                        self._expert_input(feature, expert_index, selected)
                    ).to(dtype=feature.dtype)
                    expert_outputs[expert_index] = expert_output
            shared_feature = feature + torch.sigmoid(self.shared_scale_logit) * self.shared_expert(feature)
            routed = self._aggregate_experts(
                feature,
                weights,
                expert_outputs,
                spatial_factor=spatial_factor,
            )
            output = shared_feature + gate[:, None, None, None] * self.routed_scale * routed
            if self.training:
                self._update_source_anchors(descriptor.detach(), weights.detach())
            if self.source_free_adaptation:
                self._cached_source_free_descriptor = descriptor.detach()
                self._cached_source_free_weights = weights.detach()
                self._cached_source_free_reliability = reliability.detach()
            self.last_expert_weights = weights
            self.last_expert_weights_detached = weights.detach()
            self.last_routing_confidence = confidence.detach()
            self.last_combined_reliability = reliability.detach()
            self.last_reliability_gate = gate.detach()
            self.last_active_expert_count = active_counts.detach()
            self.last_predicted_expert_benefits = (
                predicted_benefits.detach() if predicted_benefits is not None else None
            )
            self.last_gsd_risk = gsd_risk.detach() if gsd_risk is not None else None
            self.last_gsd_risk_logits = (
                gsd_risk_logits.detach() if gsd_risk_logits is not None else None
            )
            self.last_spatial_priority_logits = (
                spatial_priority_logits.detach() if spatial_priority_logits is not None else None
            )
            self.last_spatial_allocation = (
                spatial_activity.detach() if spatial_activity is not None else None
            )
            self.last_spatial_active_fraction = (
                (spatial_activity > 0.25).to(dtype=feature.dtype).mean(dim=(1, 2, 3)).detach()
                if spatial_activity is not None
                else None
            )
            self.last_spatial_allocation_mean = (
                spatial_activity.mean(dim=(1, 2, 3)).detach()
                if spatial_activity is not None
                else None
            )
            if not return_counterfactuals:
                return output
            counterfactual_features = [shared_feature]
            for count in range(1, self.top_k + 1):
                counterfactual_counts = torch.full_like(active_counts, count)
                counterfactual_weights = self._weights_for_active_counts(
                    top_values,
                    top_indices,
                    counterfactual_counts,
                )
                counterfactual_routed = self._aggregate_experts(
                    feature,
                    counterfactual_weights,
                    expert_outputs,
                    spatial_factor=spatial_factor,
                )
                counterfactual_features.append(
                    shared_feature
                    + gate[:, None, None, None] * self.routed_scale * counterfactual_routed
                )
            return {
                "output": output,
                "counterfactual_features": counterfactual_features,
                "predicted_benefits": predicted_benefits,
                "active_expert_count": active_counts,
                "gsd_risk": gsd_risk,
                "gsd_risk_logits": gsd_risk_logits,
                "spatial_priority_logits": spatial_priority_logits,
                "spatial_allocation": spatial_activity,
            }


    class ReliabilityGatedSpatialExpertHead(nn.Module):
        """Predict dense logits through a stable shared head and sparse residual experts."""

        def __init__(
            self,
            channels: int,
            classes: int,
            *,
            num_routed_experts: int = 4,
            top_k: int = 2,
            dilations: list[int] | tuple[int, ...] = (1, 2, 3, 5),
            router_temperature: float = 1.0,
            routed_scale: float = 0.5,
            reliability_threshold: float = 0.15,
            reliability_temperature: float = 0.1,
            use_reliability_gate: bool = True,
        ) -> None:
            """Build zero-initialized routed logit residuals around one shared predictor."""
            super().__init__()
            self.channels = int(channels)
            self.classes = int(classes)
            self.num_routed_experts = max(1, int(num_routed_experts))
            self.top_k = min(max(1, int(top_k)), self.num_routed_experts)
            self.router_temperature = max(float(router_temperature), 1e-4)
            self.routed_scale = float(routed_scale)
            self.reliability_threshold = float(reliability_threshold)
            self.reliability_temperature = max(float(reliability_temperature), 1e-4)
            self.use_reliability_gate = bool(use_reliability_gate)
            resolved_dilations = [max(1, int(value)) for value in dilations]
            while len(resolved_dilations) < self.num_routed_experts:
                resolved_dilations.append(resolved_dilations[-1])
            self.shared_head = nn.Conv2d(self.channels, self.classes, kernel_size=1)

            def _expert(dilation: int):
                output = nn.Conv2d(self.channels, self.classes, kernel_size=1)
                nn.init.zeros_(output.weight)
                nn.init.zeros_(output.bias)
                return nn.Sequential(
                    nn.Conv2d(
                        self.channels,
                        self.channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=self.channels,
                        bias=False,
                    ),
                    nn.GroupNorm(1, self.channels),
                    nn.GELU(),
                    output,
                )

            self.routed_experts = nn.ModuleList(
                [_expert(resolved_dilations[index]) for index in range(self.num_routed_experts)]
            )
            router_hidden = max(16, self.channels // 2)
            self.router = nn.Sequential(
                nn.LayerNorm(self.channels),
                nn.Linear(self.channels, router_hidden),
                nn.GELU(),
                nn.Linear(router_hidden, self.num_routed_experts),
            )
            nn.init.normal_(self.router[-1].weight, std=1e-3)
            nn.init.zeros_(self.router[-1].bias)
            self.last_expert_weights = None
            self.last_dense_expert_weights = None
            self.last_dense_expert_weights_detached = None
            self.last_routing_confidence = None
            self.last_combined_reliability = None
            self.last_reliability_gate = None
            self.last_active_expert_count = None

        def forward(self, feature, external_reliability=None):
            """Return shared logits plus reliability-controlled sparse expert residuals."""
            descriptor = F.adaptive_avg_pool2d(feature, 1).flatten(1)
            dense_weights = torch.softmax(self.router(descriptor) / self.router_temperature, dim=1)
            self.last_dense_expert_weights = dense_weights
            self.last_dense_expert_weights_detached = dense_weights.detach()
            entropy = -(dense_weights.clamp_min(1e-8) * dense_weights.clamp_min(1e-8).log()).sum(dim=1)
            confidence = 1.0 - entropy / max(math.log(float(self.num_routed_experts)), 1e-8)
            reliability = confidence
            if external_reliability is not None:
                external_reliability = external_reliability.to(
                    device=feature.device,
                    dtype=feature.dtype,
                ).reshape(-1)
                reliability = 0.5 * (reliability + external_reliability.clamp(0.0, 1.0))
            gate = (
                torch.sigmoid(
                    (reliability - self.reliability_threshold) / self.reliability_temperature
                )
                if self.use_reliability_gate
                else torch.ones_like(reliability)
            )
            weights = dense_weights
            if self.top_k < self.num_routed_experts:
                top_values, top_indices = torch.topk(weights, k=self.top_k, dim=1)
                sparse = torch.zeros_like(weights).scatter(1, top_indices, top_values)
                weights = sparse / sparse.sum(dim=1, keepdim=True).clamp_min(1e-8)
            routed_logits = feature.new_zeros(
                (feature.shape[0], self.classes, feature.shape[-2], feature.shape[-1])
            )
            for expert_index, expert in enumerate(self.routed_experts):
                selected = weights[:, expert_index] > 0
                if selected.any():
                    routed_logits[selected] += (
                        expert(feature[selected])
                        * weights[selected, expert_index, None, None, None]
                    )
            logits = self.shared_head(feature) + gate[:, None, None, None] * self.routed_scale * routed_logits
            self.last_expert_weights = weights
            self.last_routing_confidence = confidence.detach()
            self.last_combined_reliability = reliability.detach()
            self.last_reliability_gate = gate.detach()
            self.last_active_expert_count = (weights > 0).sum(dim=1).detach()
            return logits
else:  # pragma: no cover - torch missing
    class StageAttentionResidualFusion:  # type: ignore[override]
        """Placeholder used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Torch is required to construct StageAttentionResidualFusion.")


    class CrossStageDeltaHistoryFusion:  # type: ignore[override]
        """Placeholder used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Torch is required to construct CrossStageDeltaHistoryFusion.")


    class CrossStageDeltaHistory2d:  # type: ignore[override]
        """Placeholder used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Torch is required to construct CrossStageDeltaHistory2d.")


    class SharedRoutedExpertHead:  # type: ignore[override]
        """Placeholder used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Torch is required to construct SharedRoutedExpertHead.")


    class ReliabilityGatedSharedRoutedExpertHead:  # type: ignore[override]
        """Placeholder used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Torch is required to construct ReliabilityGatedSharedRoutedExpertHead.")


    class SharedRoutedSpatialMoE:  # type: ignore[override]
        """Placeholder used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Torch is required to construct SharedRoutedSpatialMoE.")


    class ReliabilityGatedSpatialExpertHead:  # type: ignore[override]
        """Placeholder used when torch is unavailable."""

        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Torch is required to construct ReliabilityGatedSpatialExpertHead.")

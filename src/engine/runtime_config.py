from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.paths import resolve_path


def resolve_runtime_checkpoint_paths(
    runtime_config: dict[str, Any], *, project_root: Path
) -> dict[str, Any]:
    resolved = dict(runtime_config)

    def resolve_checkpoint(value: Any) -> Any:
        if not value:
            return value
        return str(resolve_path(str(value), base_dir=project_root))

    if "classifier_checkpoint" in resolved:
        resolved["classifier_checkpoint"] = resolve_checkpoint(resolved.get("classifier_checkpoint"))
    if isinstance(resolved.get("classifier_checkpoints"), list):
        resolved["classifier_checkpoints"] = [
            resolve_checkpoint(checkpoint)
            for checkpoint in resolved["classifier_checkpoints"]
            if checkpoint
        ]
    if isinstance(resolved.get("classifier_members"), list):
        members = []
        for member in resolved["classifier_members"]:
            if not isinstance(member, dict):
                continue
            member_config = dict(member)
            if "checkpoint" in member_config:
                member_config["checkpoint"] = resolve_checkpoint(member_config.get("checkpoint"))
            members.append(member_config)
        resolved["classifier_members"] = members
    if "segmenter_checkpoint" in resolved:
        resolved["segmenter_checkpoint"] = resolve_checkpoint(resolved.get("segmenter_checkpoint"))
    if isinstance(resolved.get("segmenter_checkpoints"), list):
        resolved["segmenter_checkpoints"] = [
            resolve_checkpoint(checkpoint)
            for checkpoint in resolved["segmenter_checkpoints"]
            if checkpoint
        ]
    return resolved


@dataclass(slots=True)
class ClassifierMemberConfig:
    model: str
    checkpoint: str
    weight: float = 1.0
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        value: dict[str, Any],
        *,
        fallback_model: str,
    ) -> "ClassifierMemberConfig":
        settings = dict(value)
        model = str(settings.pop("model", fallback_model))
        checkpoint = str(settings.pop("checkpoint"))
        weight = float(settings.pop("weight", 1.0))
        return cls(model=model, checkpoint=checkpoint, weight=weight, settings=settings)

    def to_runtime_dict(self) -> dict[str, Any]:
        return {
            **self.settings,
            "model": self.model,
            "checkpoint": self.checkpoint,
            "weight": self.weight,
        }


@dataclass(slots=True)
class RuntimeConfig:
    raw: dict[str, Any]
    classifier_members: list[ClassifierMemberConfig] = field(default_factory=list)
    classifier_checkpoints: list[str] = field(default_factory=list)
    segmenter_checkpoints: list[str] = field(default_factory=list)
    default_threshold: float = 0.5
    borderline_margin: float = 0.08

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "RuntimeConfig":
        raw = dict(value or {})
        fallback_model = str(raw.get("classifier_model", "resnet18"))
        members = cls._parse_classifier_members(raw, fallback_model=fallback_model)
        classifier_checkpoints = cls._parse_checkpoint_list(
            raw,
            list_key="classifier_checkpoints",
            single_key="classifier_checkpoint",
        )
        segmenter_checkpoints = cls._parse_checkpoint_list(
            raw,
            list_key="segmenter_checkpoints",
            single_key="segmenter_checkpoint",
        )
        return cls(
            raw=raw,
            classifier_members=members,
            classifier_checkpoints=classifier_checkpoints,
            segmenter_checkpoints=segmenter_checkpoints,
            default_threshold=float(raw.get("default_threshold", 0.5)),
            borderline_margin=float(raw.get("borderline_margin", 0.08)),
        )

    @staticmethod
    def _parse_checkpoint_list(
        raw: dict[str, Any],
        *,
        list_key: str,
        single_key: str,
    ) -> list[str]:
        checkpoint_list = raw.get(list_key)
        if isinstance(checkpoint_list, list) and checkpoint_list:
            return [str(checkpoint) for checkpoint in checkpoint_list if checkpoint]
        checkpoint = raw.get(single_key)
        return [str(checkpoint)] if checkpoint else []

    @staticmethod
    def _parse_classifier_members(
        raw: dict[str, Any], *, fallback_model: str
    ) -> list[ClassifierMemberConfig]:
        configured = raw.get("classifier_members")
        if isinstance(configured, list) and configured:
            members = []
            for member in configured:
                if not isinstance(member, dict) or not member.get("checkpoint"):
                    continue
                members.append(
                    ClassifierMemberConfig.from_mapping(
                        member,
                        fallback_model=fallback_model,
                    )
                )
            if members:
                return members
        return [
            ClassifierMemberConfig(
                model=fallback_model,
                checkpoint=checkpoint,
                weight=1.0,
            )
            for checkpoint in RuntimeConfig._parse_checkpoint_list(
                raw,
                list_key="classifier_checkpoints",
                single_key="classifier_checkpoint",
            )
        ]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def member_config_value(
        self,
        member: dict[str, Any] | None,
        key: str,
        runtime_key: str,
        default: Any,
    ) -> Any:
        if member is not None:
            if key in member:
                return member[key]
            if runtime_key in member:
                return member[runtime_key]
        return self.raw.get(runtime_key, default)

    def classifier_member_dicts(self) -> list[dict[str, Any]]:
        return [member.to_runtime_dict() for member in self.classifier_members]

    def roi_enhancement_config(self) -> dict[str, Any] | None:
        config = self.raw.get("roi_enhancement")
        if not isinstance(config, dict) or not bool(config.get("enabled", False)):
            return None
        stacker = config.get("stacker")
        if not isinstance(stacker, dict):
            return None
        return config

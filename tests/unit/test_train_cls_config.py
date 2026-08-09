"""Unit tests for train cls config."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.engine.train_cls import (
    _ModelEma,
    _atomic_torch_save,
    _build_class_priors,
    _extra_model_kwargs,
    _final_classifier_metrics,
    _pairwise_auc_regularizer,
    _classification_loss,
    _count_sample_weight_hits,
    _supervised_contrastive_loss,
    _build_training_loop_config,
    _background_counterfactual_image,
    _bbox_quality_weights,
    _lesion_evidence_alignment_loss,
    _malignant_margin_loss,
    _error_aware_alignment_weights,
    _case_consistency_loss,
    _CaseGroupedBatchSampler,
    _score_checkpoint_candidate,
    run_classifier_training,
)
from src.engine import train_cls


def test_extra_model_kwargs_excludes_standard_classifier_fields() -> None:
    """Verify extra model kwargs excludes standard classifier fields."""
    assert _extra_model_kwargs(
        {
            "name": "convnext_small",
            "pretrained": True,
            "in_chans": 3,
            "num_classes": 2,
            "drop_path_rate": 0.2,
        }
    ) == {"drop_path_rate": 0.2}


def test_checkpoint_scoring_reuses_validation_probabilities() -> None:
    """Verify checkpoint scoring reuses validation probabilities."""
    metrics, score = _score_checkpoint_candidate(
        epoch_metrics={"auc": 0.9, "sensitivity": 0.5, "specificity": 0.5},
        y_true=[0, 1],
        malignant_probabilities=[0.2, 0.8],
        checkpoint_strategy="selected_threshold",
        selection_threshold=0.5,
        sensitivity_weight=0.0,
        min_specificity=0.0,
    )

    assert metrics["auc"] == 0.9
    assert metrics["sensitivity"] == 1.0
    assert score > 0.0


def test_checkpoint_scoring_can_select_by_f1() -> None:
    """Verify best_f1 checkpoint strategy scores by validation F1."""
    metrics, score = _score_checkpoint_candidate(
        epoch_metrics={
            "auc": 0.99,
            "sensitivity": 0.5,
            "specificity": 0.5,
            "f1_score": 0.42,
        },
        y_true=[0, 1],
        malignant_probabilities=[0.2, 0.8],
        checkpoint_strategy="best_f1",
        selection_threshold=0.5,
        sensitivity_weight=0.0,
        min_specificity=0.0,
    )

    assert metrics["auc"] == 0.99
    assert score == 0.42


def test_last_checkpoint_metrics_reuse_final_epoch_report(monkeypatch) -> None:
    """Verify last checkpoint metrics reuse final epoch report."""
    def _fail_evaluate_model(*_args, **_kwargs):
        """Return fail evaluate model."""
        raise AssertionError("last checkpoint should reuse final epoch validation metrics")

    monkeypatch.setattr(train_cls, "_evaluate_model", _fail_evaluate_model)

    metrics = _final_classifier_metrics(
        model=object(),
        val_loader=object(),
        device="cpu",
        checkpoint_strategy="last",
        epoch_reports=[{"metrics": {"auc": 0.75, "threshold": 0.5}}],
    )

    assert metrics == {"auc": 0.75, "threshold": 0.5}


def test_non_last_checkpoint_metrics_revalidate_selected_state(monkeypatch) -> None:
    """Verify non last checkpoint metrics revalidate selected state."""
    monkeypatch.setattr(
        train_cls,
        "_evaluate_model",
        lambda *_args, **_kwargs: {"auc": 0.91, "threshold": 0.5},
    )

    metrics = _final_classifier_metrics(
        model=object(),
        val_loader=object(),
        device="cpu",
        checkpoint_strategy="youden",
        epoch_reports=[{"metrics": {"auc": 0.75, "threshold": 0.5}}],
    )

    assert metrics == {"auc": 0.91, "threshold": 0.5}


def test_atomic_torch_save_replaces_destination(tmp_path: Path) -> None:
    """Verify atomic torch save replaces destination."""
    if train_cls.torch is None:
        return
    destination = tmp_path / "model.pt"

    _atomic_torch_save({"value": 1}, destination)
    _atomic_torch_save({"value": 2}, destination)

    assert destination.exists()
    assert not (tmp_path / ".model.pt.tmp").exists()


def test_pairwise_auc_regularizer_prefers_correct_ranking() -> None:
    """Verify the pairwise AUC surrogate is smaller for correctly ranked pairs."""
    if train_cls.torch is None:
        return
    torch = train_cls.torch
    labels = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    good_logits = torch.tensor(
        [[0.1, 2.5], [0.2, 2.0], [2.2, 0.2], [1.9, 0.1]],
        dtype=torch.float32,
    )
    bad_logits = torch.tensor(
        [[2.0, 0.1], [1.8, 0.2], [0.1, 2.0], [0.2, 1.9]],
        dtype=torch.float32,
    )

    good_loss = _pairwise_auc_regularizer(
        good_logits,
        labels,
        sample_weights=None,
        margin=0.0,
    )
    bad_loss = _pairwise_auc_regularizer(
        bad_logits,
        labels,
        sample_weights=None,
        margin=0.0,
    )

    assert float(good_loss.item()) < float(bad_loss.item())


def test_build_training_loop_config_reads_pairwise_auc_settings() -> None:
    """Verify pairwise AUC settings are normalized into the loop config."""
    loop_config = _build_training_loop_config(
        training_cfg={
            "epochs": 3,
            "pairwise_auc_weight": 0.2,
            "pairwise_auc_margin": 0.1,
        },
        epochs_override=None,
        class_weights=None,
        class_priors=None,
        sample_weights={},
    )

    assert loop_config.pairwise_auc_weight == 0.2
    assert loop_config.pairwise_auc_margin == 0.1


def test_build_training_loop_config_reads_supcon_settings() -> None:
    """Verify supervised contrastive settings are normalized into the loop config."""
    loop_config = _build_training_loop_config(
        training_cfg={
            "epochs": 3,
            "supcon_weight": 0.1,
            "supcon_temperature": 0.2,
        },
        epochs_override=None,
        class_weights=None,
        class_priors=None,
        sample_weights={},
    )

    assert loop_config.supcon_weight == 0.1
    assert loop_config.supcon_temperature == 0.2


def test_build_training_loop_config_reads_local_evidence_classification_weight() -> None:
    """Verify label-aware lesion supervision weight is normalized for the epoch loop."""
    loop_config = _build_training_loop_config(
        training_cfg={
            "epochs": 3,
            "lesion_evidence": {"local_classification_weight": 0.1},
        },
        epochs_override=None,
        class_weights=None,
        class_priors=None,
        sample_weights={},
    )

    assert loop_config.local_evidence_classification_weight == 0.1


def test_build_training_loop_config_reads_quality_and_background_settings() -> None:
    """Verify evidence-quality and counterfactual settings reach the epoch loop unchanged."""
    loop_config = _build_training_loop_config(
        training_cfg={
            "epochs": 3,
            "lesion_evidence": {
                "alignment_weight": 0.25,
                "quality_weighting": {"enabled": True},
            },
            "background_consistency": {"weight": 0.05, "probability": 0.5},
        },
        epochs_override=None,
        class_weights=None,
        class_priors=None,
        sample_weights={},
        bbox_quality_stats={"area_q05": 0.1, "area_q95": 0.6, "log_aspect_q95": 1.2},
    )

    assert loop_config.lesion_evidence_quality_weighting is True
    assert loop_config.lesion_evidence_quality_stats["area_q05"] == 0.1
    assert loop_config.background_consistency_weight == 0.05
    assert loop_config.background_consistency_probability == 0.5


def test_case_consistency_matches_same_case_predictions_and_reports_coverage() -> None:
    """Verify paired-case KL is active only when the batch contains a pair."""
    if train_cls.torch is None:
        return
    torch = train_cls.torch
    logits = torch.tensor(
        [[0.0, 2.0], [0.0, 2.0], [2.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    loss, coverage, pairs = _case_consistency_loss(
        logits,
        ["case-1", "case-1", "case-2"],
        temperature=0.5,
    )
    assert float(loss.item()) < 1e-6
    assert coverage == pytest.approx(2.0 / 3.0)
    assert pairs == 1


def test_case_grouped_batch_sampler_keeps_case_views_together() -> None:
    manifest = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d", "e"],
            "case_id": ["1", "1", "2", "3", "3"],
        }
    )
    sampler = _CaseGroupedBatchSampler(manifest, batch_size=4, seed=42)
    batches = list(iter(sampler))
    location = {}
    for batch in batches:
        case_ids = manifest.iloc[batch]["case_id"].astype(str).tolist()
        for case_id in set(case_ids):
            positions = [index for index, value in enumerate(case_ids) if value == case_id]
            assert len(positions) == len(set(positions))
            location.setdefault(case_id, set()).add(tuple(batch))
    assert len(location["1"]) == 1
    assert len(location["3"]) == 1
    flattened = [index for batch in batches for index in batch]
    assert sorted(flattened) == list(range(len(manifest)))


def test_build_training_loop_config_reads_error_aware_alignment_settings() -> None:
    """Verify hard-example alignment settings are normalized without changing inference."""
    loop_config = _build_training_loop_config(
        training_cfg={
            "epochs": 3,
            "lesion_evidence": {
                "error_aware_weighting": {"enabled": True, "scale": 0.5},
            },
        },
        epochs_override=None,
        class_weights=None,
        class_priors=None,
        sample_weights={},
    )

    assert loop_config.lesion_evidence_error_aware_weighting is True
    assert loop_config.lesion_evidence_error_weight_scale == 0.5


def test_build_training_loop_config_reads_case_consistency_settings() -> None:
    """Verify case-level consistency settings are normalized into the loop config."""
    loop_config = _build_training_loop_config(
        training_cfg={
            "epochs": 3,
            "case_consistency": {"weight": 0.02, "temperature": 0.5},
        },
        epochs_override=None,
        class_weights=None,
        class_priors=None,
        sample_weights={},
    )

    assert loop_config.case_consistency_weight == 0.02
    assert loop_config.case_consistency_temperature == 0.5


def test_bbox_quality_weights_respect_bounds_and_missing_labels() -> None:
    """Verify fixed quality weights remain bounded and ignore samples lacking BBOX metadata."""
    if train_cls.torch is None:
        return
    torch = train_cls.torch
    boxes = torch.tensor(
        [[0.2, 0.2, 0.6, 0.6], [0.0, 0.0, 0.95, 0.05], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    valid = torch.tensor([1.0, 1.0, 0.0])
    weights = _bbox_quality_weights(
        boxes,
        valid,
        {"area_q05": 0.1, "area_q95": 0.5, "log_aspect_q95": 1.5},
    )

    assert torch.all((weights[:2] >= 0.75) & (weights[:2] <= 1.25))
    assert float(weights[-1]) == 0.0


def test_error_aware_alignment_weights_are_bounded_and_focus_low_true_probability() -> None:
    """Verify hard-example evidence weights are bounded and normalized on valid boxes."""
    if train_cls.torch is None:
        return
    torch = train_cls.torch
    logits = torch.tensor(
        [[0.0, 4.0], [0.0, 0.0], [4.0, 0.0], [0.0, 0.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([1, 1, 0, 0], dtype=torch.long)
    valid = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    weights = _error_aware_alignment_weights(
        logits,
        labels,
        valid,
        scale=0.5,
    )

    assert torch.all((weights[:3] >= 0.75) & (weights[:3] <= 1.25))
    assert float(weights[-1]) == 0.0
    assert float(weights[1]) > float(weights[0])
    assert float(weights[:3].mean()) == pytest.approx(1.0, abs=1e-6)


def test_alignment_error_aware_mode_requires_same_forward_labels() -> None:
    """Verify the error-aware option fails closed when labels/logits are unavailable."""
    if train_cls.torch is None:
        return
    torch = train_cls.torch

    class _Model:
        last_evidence_maps = [torch.zeros((1, 1, 7, 7), requires_grad=True)]
        last_logits = None

    images = {
        "bbox": torch.tensor([[0.2, 0.2, 0.8, 0.8]], dtype=torch.float32),
        "bbox_valid": torch.tensor([1.0]),
    }
    with pytest.raises(ValueError, match="error_aware_weighting"):
        _lesion_evidence_alignment_loss(
            _Model(),
            images,
            labels=torch.tensor([1]),
            error_aware_weighting=True,
        )


def test_alignment_supports_multiple_evidence_maps_and_missing_bboxes() -> None:
    """Verify multi-resolution alignment averages maps and skips invalid BBOX rows safely."""
    if train_cls.torch is None:
        return
    torch = train_cls.torch

    class _Model:
        last_evidence_maps = [
            torch.zeros((2, 1, 14, 14), requires_grad=True),
            torch.zeros((2, 1, 7, 7), requires_grad=True),
        ]

    model = _Model()
    images = {
        "bbox": torch.tensor([[0.2, 0.2, 0.8, 0.8], [0.0, 0.0, 0.0, 0.0]]),
        "bbox_valid": torch.tensor([1.0, 0.0]),
    }
    loss, coverage = _lesion_evidence_alignment_loss(model, images)

    assert loss is not None
    assert float(loss.detach()) > 0.0
    assert coverage == 0.5
    assert tuple(model.last_alignment_quality_weight.shape) == (2,)


def test_perilesional_alignment_preserves_core_loss_at_zero_weight() -> None:
    """Verify zero context weight preserves v1a alignment and positive weight adds ring mass."""
    if train_cls.torch is None:
        return
    torch = train_cls.torch

    class _Model:
        last_evidence_maps = [torch.zeros((1, 1, 7, 7), requires_grad=True)]

    images = {
        "bbox": torch.tensor([[0.3, 0.3, 0.5, 0.5]], dtype=torch.float32),
        "bbox_valid": torch.tensor([1.0]),
    }
    core_loss, _ = _lesion_evidence_alignment_loss(_Model(), images, margin=0.08)
    zero_loss, _ = _lesion_evidence_alignment_loss(
        _Model(), images, margin=0.08, perilesional_context_weight=0.0
    )
    soft_loss, _ = _lesion_evidence_alignment_loss(
        _Model(), images, margin=0.08, perilesional_context_weight=0.15
    )

    assert torch.allclose(core_loss, zero_loss)
    assert float(soft_loss.detach()) < float(core_loss.detach())


def test_background_counterfactual_preserves_lesion_pixels() -> None:
    """Verify background augmentation cannot alter the original or flipped lesion region."""
    if train_cls.torch is None:
        return
    torch = train_cls.torch
    torch.manual_seed(7)
    image = torch.linspace(-1.0, 1.0, 3 * 16 * 16, dtype=torch.float32).reshape(1, 3, 16, 16)
    images = {
        "image": image.clone(),
        "bbox": torch.tensor([[0.25, 0.25, 0.50, 0.50]], dtype=torch.float32),
        "bbox_valid": torch.tensor([1.0]),
    }
    perturbed, coverage = _background_counterfactual_image(images, probability=1.0)

    assert perturbed is not None
    assert coverage > 0.0
    assert torch.equal(perturbed[:, :, 4:8, 4:8], image[:, :, 4:8, 4:8])
    assert torch.equal(perturbed[:, :, 4:8, 8:12], image[:, :, 4:8, 8:12])
    assert not torch.equal(perturbed, image)


def test_build_training_loop_config_reads_sam_settings() -> None:
    """Verify SAM settings are normalized into the loop config."""
    loop_config = _build_training_loop_config(
        training_cfg={
            "epochs": 3,
            "optimizer": "sam",
            "sam_rho": 0.05,
            "sam_adaptive": True,
        },
        epochs_override=None,
        class_weights=None,
        class_priors=None,
        sample_weights={},
    )

    assert loop_config.use_sam is True
    assert loop_config.sam_rho == 0.05
    assert loop_config.sam_adaptive is True


def test_build_training_loop_config_reads_early_stopping_settings() -> None:
    """Verify early stopping settings are preserved for the training loop."""
    default_config = _build_training_loop_config(
        training_cfg={"epochs": 3},
        epochs_override=None,
        class_weights=None,
        class_priors=None,
        sample_weights={},
    )
    enabled_config = _build_training_loop_config(
        training_cfg={
            "epochs": 18,
            "early_stopping": {
                "enabled": True,
                "patience": 5,
                "min_delta": 0.001,
                "monitor": "selection_score",
            },
        },
        epochs_override=None,
        class_weights=None,
        class_priors=None,
        sample_weights={},
    )

    assert default_config.early_stopping_cfg["enabled"] is False
    assert enabled_config.early_stopping_cfg["enabled"] is True
    assert enabled_config.early_stopping_cfg["patience"] == 5
    assert enabled_config.early_stopping_cfg["monitor"] == "selection_score"


def test_build_class_priors_returns_empirical_distribution_for_balanced_softmax() -> None:
    """Verify balanced_softmax reads empirical class priors from the fold manifest."""
    if train_cls.torch is None:
        return
    manifest = pd.DataFrame(
        [
            {"pathology_label": "benign"},
            {"pathology_label": "benign"},
            {"pathology_label": "benign"},
            {"pathology_label": "malignant"},
        ]
    )

    priors = _build_class_priors(
        manifest,
        {"loss": "balanced_softmax"},
        device="cpu",
    )

    assert priors is not None
    assert train_cls.torch.allclose(
        priors,
        train_cls.torch.tensor([0.75, 0.25], dtype=train_cls.torch.float32),
        atol=1e-6,
    )


def test_balanced_softmax_reduces_loss_for_majority_class_logits() -> None:
    """Verify balanced_softmax applies class-frequency correction through log priors."""
    if train_cls.torch is None:
        return
    torch = train_cls.torch
    logits = torch.tensor([[0.1, 0.1]], dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.long)
    priors = torch.tensor([0.75, 0.25], dtype=torch.float32)

    ce_loss = _classification_loss(
        logits,
        labels,
        class_weights=None,
        class_priors=None,
        sample_weights=None,
        label_smoothing=0.0,
        loss_name="cross_entropy",
        focal_gamma=2.0,
        balanced_softmax_tau=1.0,
    )
    bs_loss = _classification_loss(
        logits,
        labels,
        class_weights=None,
        class_priors=priors,
        sample_weights=None,
        label_smoothing=0.0,
        loss_name="balanced_softmax",
        focal_gamma=2.0,
        balanced_softmax_tau=1.0,
    )

    assert float(bs_loss.item()) > float(ce_loss.item())


def test_balanced_softmax_tau_controls_adjustment_strength() -> None:
    """Verify smaller tau weakens the logit adjustment effect."""
    if train_cls.torch is None:
        return
    torch = train_cls.torch
    logits = torch.tensor([[0.1, 0.1]], dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.long)
    priors = torch.tensor([0.75, 0.25], dtype=torch.float32)

    mild_loss = _classification_loss(
        logits,
        labels,
        class_weights=None,
        class_priors=priors,
        sample_weights=None,
        label_smoothing=0.0,
        loss_name="balanced_softmax",
        focal_gamma=2.0,
        balanced_softmax_tau=0.4,
    )
    strong_loss = _classification_loss(
        logits,
        labels,
        class_weights=None,
        class_priors=priors,
        sample_weights=None,
        label_smoothing=0.0,
        loss_name="balanced_softmax",
        focal_gamma=2.0,
        balanced_softmax_tau=1.0,
    )

    assert float(mild_loss.item()) < float(strong_loss.item())


def test_count_sample_weight_hits_counts_non_default_training_matches() -> None:
    """Verify sample weight hit counting only counts train samples with non-default weights."""
    manifest = pd.DataFrame(
        [
            {"sample_id": "a"},
            {"sample_id": "b"},
            {"sample_id": "c"},
        ]
    )

    hits = _count_sample_weight_hits(
        manifest,
        {
            "a": 1.0,
            "b": 1.2,
            "outside": 1.5,
        },
    )

    assert hits == 1


def test_supervised_contrastive_loss_prefers_separable_embeddings() -> None:
    """Verify the supervised contrastive term is lower for class-clustered embeddings."""
    if train_cls.torch is None:
        return
    torch = train_cls.torch
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    good_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    bad_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    good_loss = _supervised_contrastive_loss(good_embeddings, labels, temperature=0.1)
    bad_loss = _supervised_contrastive_loss(bad_embeddings, labels, temperature=0.1)

    assert float(good_loss.item()) < float(bad_loss.item())


def test_malignant_margin_only_penalizes_insufficient_positive_logit_gap() -> None:
    """Verify the margin term has no loss once malignant logits exceed the target gap."""
    if train_cls.torch is None:
        return
    torch = train_cls.torch
    labels = torch.tensor([1, 0, 1], dtype=torch.long)
    logits = torch.tensor([[0.0, 0.20], [0.0, 4.0], [0.0, 0.02]], dtype=torch.float32)
    loss = _malignant_margin_loss(logits, labels, margin=0.10)
    assert float(loss) == pytest.approx(0.04, abs=1e-6)


def test_model_ema_updates_toward_latest_weights() -> None:
    """Verify EMA state moves toward the latest model weights."""
    if train_cls.torch is None:
        return
    torch = train_cls.torch
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = _ModelEma(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)
    assert float(ema.state_dict["weight"].mean().item()) == 2.0


def test_run_classifier_training_smoke_uses_loop_config_outputs(tmp_path: Path, monkeypatch) -> None:
    """Verify run classifier training smoke uses loop config outputs."""
    if train_cls.torch is None:
        return
    torch = train_cls.torch
    manifest = pd.DataFrame(
        [
            {"sample_id": "a", "case_id": "c1", "pathology_label": "benign"},
            {"sample_id": "b", "case_id": "c2", "pathology_label": "malignant"},
            {"sample_id": "c", "case_id": "c3", "pathology_label": "benign"},
            {"sample_id": "d", "case_id": "c4", "pathology_label": "malignant"},
        ]
    )

    class _Dataset(torch.utils.data.Dataset):
        """Represent Dataset for this module."""
        def __init__(self, frame, **_kwargs) -> None:
            """Initialize this lightweight test helper."""
            self.frame = frame.reset_index(drop=True)

        def __len__(self) -> int:
            """Return the number of samples in this helper dataset."""
            return len(self.frame)

        def __getitem__(self, index: int):
            """Return one sample from this helper dataset."""
            row = self.frame.iloc[index]
            label = 1 if row["pathology_label"] == "malignant" else 0
            return {
                "image": torch.full((3, 8, 8), float(label)),
                "label": torch.tensor(label, dtype=torch.long),
                "sample_id": row["sample_id"],
            }

    class _TinyClassifier(torch.nn.Module):
        """Represent TinyClassifier for this module."""
        def __init__(self) -> None:
            """Initialize this lightweight test helper."""
            super().__init__()
            self.fc = torch.nn.Linear(3 * 8 * 8, 2)

        def forward(self, images):
            """Run forward."""
            return self.fc(images.flatten(1))

    class _Paths:
        """Represent Paths for this module."""
        project_root = tmp_path
        busbra_root = tmp_path / "unused"
        checkpoints_root = tmp_path / "checkpoints"
        reports_root = tmp_path / "reports"

    config_path = tmp_path / "config.yml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    monkeypatch.setattr(
        train_cls,
        "load_project_config",
        lambda _path: ({"device": "cpu", "training": {"epochs": 1}, "output": {}}, _Paths()),
    )
    monkeypatch.setattr(train_cls, "load_busbra_manifest", lambda _root: manifest)
    monkeypatch.setattr(train_cls, "_prepare_fold_manifests", lambda **_kwargs: (manifest.iloc[:2], manifest.iloc[2:]))
    monkeypatch.setattr(train_cls, "BUSBRAClassificationDataset", _Dataset)
    monkeypatch.setattr(train_cls, "_build_classifier_model_and_transforms", lambda **_kwargs: (_TinyClassifier(), 8, None, None, {"image_size": 8}))

    report = run_classifier_training(config_path, fold=1)

    assert report["checkpoint_strategy"] == "last"
    assert report["scheduler"] == {}
    assert report["min_specificity"] == 0.0
    assert report["pairwise_auc_weight"] == 0.0
    assert report["pairwise_auc_margin"] == 0.0
    assert report["use_ema"] is False
    assert report["ema_decay"] is None
    assert report["early_stopping"]["enabled"] is False
    assert report["stopped_epoch"] is None
    assert len(report["epoch_reports"]) == 1


def test_dualview_batch_routing_calls_model_with_named_inputs() -> None:
    """Verify train_cls routes dual-view batches through named model inputs."""
    if train_cls.torch is None:
        return
    torch = train_cls.torch
    batch = {
        "image_full": torch.zeros((2, 3, 8, 8), dtype=torch.float32),
        "image_roi": torch.ones((2, 3, 8, 8), dtype=torch.float32),
        "roi_descriptor": torch.zeros((2, 14), dtype=torch.float32),
        "roi_valid": torch.ones((2,), dtype=torch.float32),
        "label": torch.tensor([0, 1], dtype=torch.long),
        "sample_id": ["a", "b"],
    }

    class _DualViewModel(torch.nn.Module):
        """Small dual-view model for batch routing tests."""

        def forward_with_embedding(self, image_full, image_roi, roi_descriptor, labels=None, **_kwargs):
            """Return logits based on named inputs."""
            assert tuple(image_full.shape) == (2, 3, 8, 8)
            assert tuple(image_roi.shape) == (2, 3, 8, 8)
            assert tuple(roi_descriptor.shape) == (2, 14)
            assert labels.tolist() == [0, 1]
            logits = torch.stack(
                [image_full.mean((1, 2, 3)), image_roi.mean((1, 2, 3))],
                dim=1,
            )
            return logits, roi_descriptor

    images, labels, weights = train_cls._prepare_classifier_batch(batch, {}, device="cpu")
    logits, embeddings = train_cls._model_logits_and_embedding(_DualViewModel(), images, labels)

    assert isinstance(images, dict)
    assert weights is None
    assert tuple(logits.shape) == (2, 2)
    assert tuple(embeddings.shape) == (2, 14)


def test_teacher_constraint_backpropagates_only_for_reliable_teacher_samples() -> None:
    """Ensure frozen teachers constrain student logits without entering inference state."""
    if train_cls.torch is None:
        return
    torch = train_cls.torch

    class _Teacher(torch.nn.Module):
        """Return fixed two-class logits for one reliability-gating test."""

        def __init__(self, logits) -> None:
            """Store fixed logits without trainable parameters."""
            super().__init__()
            self.register_buffer("fixed_logits", torch.tensor(logits, dtype=torch.float32))

        def forward(self, images):
            """Expand fixed logits to the current batch size."""
            return self.fixed_logits[: images.shape[0]]

    student_logits = torch.zeros((2, 2), dtype=torch.float32, requires_grad=True)
    images = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
    first_teacher = _Teacher([[4.0, 0.0], [3.0, 0.0]])
    second_teacher = _Teacher([[3.5, 0.0], [0.0, 3.0]])

    loss, coverage = train_cls._teacher_constraint_loss(
        student_logits,
        images,
        [first_teacher, second_teacher],
        [0.5, 0.5],
        temperature=2.0,
        min_confidence=0.7,
        max_disagreement=0.12,
    )
    loss.backward()

    assert 0.45 <= float(coverage.item()) <= 0.55
    assert float(student_logits.grad[0].abs().sum().item()) > 0.0
    assert float(student_logits.grad[1].abs().sum().item()) == 0.0

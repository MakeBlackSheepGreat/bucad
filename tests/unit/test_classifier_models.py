"""Unit tests for classifier models."""

from __future__ import annotations

import pytest

from src.models import classifier
from src.models.classifier import create_classifier, load_classifier, resolve_gradcam_target_layer


@pytest.mark.skipif(classifier.torch is None, reason="torch is not installed")
def test_basic_cnn_classifier_outputs_two_logits() -> None:
    """Verify basic cnn classifier outputs two logits."""
    model = create_classifier("basic_cnn", pretrained=False, in_chans=3, num_classes=2)
    batch = classifier.torch.zeros((2, 3, 32, 32), dtype=classifier.torch.float32)

    logits = model(batch)

    assert tuple(logits.shape) == (2, 2)


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
def test_efficientnetv2_s_classifier_can_be_constructed_without_pretrained_weights() -> None:
    """Verify efficientnetv2 s classifier can be constructed without pretrained weights."""
    model = create_classifier("tf_efficientnetv2_s", pretrained=False, in_chans=3, num_classes=2)

    assert model is not None


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
def test_timm_classifier_accepts_regularization_kwargs() -> None:
    """Verify timm classifier accepts regularization kwargs."""
    model = create_classifier(
        "convnext_tiny",
        pretrained=False,
        in_chans=3,
        num_classes=2,
        drop_path_rate=0.1,
    )

    assert model is not None


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
def test_sonoglore_convnext_tiny_outputs_two_logits_and_exposes_metadata() -> None:
    """Verify SonoGloReNet tiny constructs, runs, and exposes pretrained metadata."""
    model = create_classifier(
        "sonoglore_convnext_tiny",
        pretrained=False,
        in_chans=3,
        num_classes=2,
    )
    batch = classifier.torch.zeros((2, 3, 224, 224), dtype=classifier.torch.float32)

    logits = model(batch)

    assert tuple(logits.shape) == (2, 2)
    assert isinstance(getattr(model, "pretrained_cfg", None), dict)
    assert tuple(model.active_stage_indices) == (2, 3, 4)
    assert model.use_stage4_attention is True
    assert model.stage4_attention_type == "mhsa"
    assert resolve_gradcam_target_layer(model) is model.gradcam_layer


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
@pytest.mark.parametrize(
    ("active_stage_indices", "use_stage4_attention"),
    [
        ([2, 3, 4], False),
        ([3, 4], False),
        ([4], False),
    ],
)
def test_sonoglore_ablation_variants_output_two_logits(
    active_stage_indices,
    use_stage4_attention,
) -> None:
    """Verify ablation variants keep the expected binary output shape."""
    model = create_classifier(
        "sonoglore_convnext_tiny",
        pretrained=False,
        in_chans=3,
        num_classes=2,
        active_stage_indices=active_stage_indices,
        use_stage4_attention=use_stage4_attention,
    )
    batch = classifier.torch.zeros((2, 3, 224, 224), dtype=classifier.torch.float32)

    logits = model(batch)

    assert tuple(logits.shape) == (2, 2)
    assert tuple(model.active_stage_indices) == tuple(active_stage_indices)
    assert model.use_stage4_attention is use_stage4_attention
    assert resolve_gradcam_target_layer(model) is model.gradcam_layer


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
def test_sonoglore_v2_grn_and_scale_gate_outputs_two_logits() -> None:
    """Verify SonoGloReNet v2-style options keep construction and output stable."""
    model = create_classifier(
        "sonoglore_convnext_tiny",
        pretrained=False,
        in_chans=3,
        num_classes=2,
        active_stage_indices=[2, 3, 4],
        use_stage4_attention=False,
        use_projection_grn=True,
        use_scale_gate=True,
        scale_gate_hidden_dim=128,
    )
    batch = classifier.torch.zeros((2, 3, 224, 224), dtype=classifier.torch.float32)

    logits = model(batch)

    assert tuple(logits.shape) == (2, 2)
    assert model.use_projection_grn is True
    assert model.scale_gate is not None
    assert model.last_scale_gate_weights is not None
    assert tuple(model.last_scale_gate_weights.shape) == (2, 3)
    assert resolve_gradcam_target_layer(model) is model.gradcam_layer


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
def test_sonoglore_stage4_semantic_scale_gate_outputs_two_logits() -> None:
    """Verify stage4-conditioned scale gating constructs and runs."""
    model = create_classifier(
        "sonoglore_convnext_tiny",
        pretrained=False,
        in_chans=3,
        num_classes=2,
        active_stage_indices=[2, 3, 4],
        use_stage4_attention=False,
        use_projection_grn=True,
        use_projection_eca=True,
        use_scale_gate=True,
        scale_gate_mode="stage4",
        scale_gate_hidden_dim=128,
    )
    batch = classifier.torch.zeros((2, 3, 224, 224), dtype=classifier.torch.float32)

    logits = model(batch)

    assert tuple(logits.shape) == (2, 2)
    assert model.scale_gate is not None
    assert model.scale_gate_mode == "stage4"
    assert model.last_scale_gate_weights is not None
    assert tuple(model.last_scale_gate_weights.shape) == (2, 3)
    assert resolve_gradcam_target_layer(model) is model.gradcam_layer


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
def test_sonoglore_stage4_residual_scale_gate_outputs_two_logits() -> None:
    """Verify conservative residual stage4 scale gating constructs and stays identity-centered."""
    model = create_classifier(
        "sonoglore_convnext_tiny",
        pretrained=False,
        in_chans=3,
        num_classes=2,
        active_stage_indices=[2, 3, 4],
        use_stage4_attention=False,
        use_projection_grn=True,
        use_projection_eca=True,
        use_scale_gate=True,
        scale_gate_mode="stage4_residual",
        scale_gate_hidden_dim=128,
        scale_gate_delta=0.15,
        scale_gate_anchor_last=True,
    )
    batch = classifier.torch.zeros((2, 3, 224, 224), dtype=classifier.torch.float32)

    logits = model(batch)

    assert tuple(logits.shape) == (2, 2)
    assert model.scale_gate is not None
    assert model.scale_gate_mode == "stage4_residual"
    assert model.last_scale_gate_weights is not None
    assert tuple(model.last_scale_gate_weights.shape) == (2, 3)
    assert classifier.torch.allclose(
        model.last_scale_gate_weights,
        classifier.torch.ones_like(model.last_scale_gate_weights),
        atol=1e-6,
    )
    assert resolve_gradcam_target_layer(model) is model.gradcam_layer


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
def test_sonoglore_grn_eca_outputs_two_logits() -> None:
    """Verify the GRN plus ECA variant constructs and runs."""
    model = create_classifier(
        "sonoglore_convnext_tiny",
        pretrained=False,
        in_chans=3,
        num_classes=2,
        active_stage_indices=[2, 3, 4],
        use_stage4_attention=False,
        use_projection_grn=True,
        use_projection_eca=True,
    )
    batch = classifier.torch.zeros((2, 3, 224, 224), dtype=classifier.torch.float32)

    logits = model(batch)

    assert tuple(logits.shape) == (2, 2)
    assert model.use_projection_grn is True
    assert model.use_projection_eca is True
    assert model.scale_gate is None
    assert resolve_gradcam_target_layer(model) is model.gradcam_layer


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
@pytest.mark.parametrize("stage4_attention_type", ["coordatt", "gc", "lka"])
def test_sonoglore_alternative_stage4_attention_outputs_two_logits(stage4_attention_type) -> None:
    """Verify alternative stage4 attention variants construct and run."""
    model = create_classifier(
        "sonoglore_convnext_tiny",
        pretrained=False,
        in_chans=3,
        num_classes=2,
        use_stage4_attention=True,
        stage4_attention_type=stage4_attention_type,
        use_projection_grn=True,
        use_projection_eca=True,
    )
    batch = classifier.torch.zeros((2, 3, 224, 224), dtype=classifier.torch.float32)

    logits = model(batch)

    assert tuple(logits.shape) == (2, 2)
    assert model.stage4_attention_type == stage4_attention_type
    assert resolve_gradcam_target_layer(model) is model.gradcam_layer


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
def test_sonoglore_asymmetric_projection_and_stage_weights_output_two_logits() -> None:
    """Verify asymmetric projection dimensions and stage weights remain compatible."""
    model = create_classifier(
        "sonoglore_convnext_tiny",
        pretrained=False,
        in_chans=3,
        num_classes=2,
        active_stage_indices=[2, 3, 4],
        stage_proj_dims=[160, 256, 320],
        stage_fusion_weights=[0.55, 1.0, 1.05],
        use_stage4_attention=False,
        use_projection_grn=True,
        use_projection_eca=True,
    )
    batch = classifier.torch.zeros((2, 3, 224, 224), dtype=classifier.torch.float32)

    logits = model(batch)

    assert tuple(logits.shape) == (2, 2)
    assert tuple(model.stage_proj_dims) == (160, 256, 320)
    assert tuple(model.stage_fusion_weights) == (0.55, 1.0, 1.05)
    assert model.head_fc1.in_features == 160 + 256 + 320
    assert resolve_gradcam_target_layer(model) is model.gradcam_layer


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
def test_sonoglore_arc_margin_head_outputs_two_logits() -> None:
    """Verify the optional ArcMargin head constructs and supports label-aware forward."""
    model = create_classifier(
        "sonoglore_convnext_tiny",
        pretrained=False,
        in_chans=3,
        num_classes=2,
        active_stage_indices=[2, 3, 4],
        use_stage4_attention=False,
        use_projection_grn=True,
        use_projection_eca=True,
        classifier_head_type="arc_margin",
        classifier_head_scale=16.0,
        classifier_head_margin=0.15,
    )
    batch = classifier.torch.zeros((2, 3, 224, 224), dtype=classifier.torch.float32)
    labels = classifier.torch.tensor([0, 1], dtype=classifier.torch.long)

    logits = model(batch, labels=labels)
    infer_logits = model(batch)

    assert tuple(logits.shape) == (2, 2)
    assert tuple(infer_logits.shape) == (2, 2)
    assert model.classifier_head_type == "arc_margin"
    assert resolve_gradcam_target_layer(model) is model.gradcam_layer


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
def test_sonoglore_forward_with_embedding_returns_logits_and_embedding() -> None:
    """Verify SonoGloReNet can expose its fused embedding for auxiliary losses."""
    model = create_classifier(
        "sonoglore_convnext_tiny",
        pretrained=False,
        in_chans=3,
        num_classes=2,
        active_stage_indices=[2, 3, 4],
        use_stage4_attention=False,
        use_projection_grn=True,
        use_projection_eca=True,
    )
    batch = classifier.torch.zeros((2, 3, 224, 224), dtype=classifier.torch.float32)

    logits, embedding = model.forward_with_embedding(batch)

    assert tuple(logits.shape) == (2, 2)
    assert embedding.ndim == 2
    assert embedding.shape[0] == 2
    assert embedding.shape[1] == 256


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
def test_sonoglore_mixstyle_outputs_two_logits() -> None:
    """Verify optional MixStyle stages construct and keep binary output stable."""
    model = create_classifier(
        "sonoglore_convnext_tiny",
        pretrained=False,
        in_chans=3,
        num_classes=2,
        active_stage_indices=[2, 3, 4],
        use_stage4_attention=False,
        use_projection_grn=True,
        use_projection_eca=True,
        use_mixstyle=True,
        mixstyle_stages=[2, 3],
        mixstyle_probability=1.0,
        mixstyle_alpha=0.1,
    )
    model.train()
    batch = classifier.torch.zeros((2, 3, 224, 224), dtype=classifier.torch.float32)

    logits = model(batch)

    assert tuple(logits.shape) == (2, 2)
    assert model.use_mixstyle is True
    assert tuple(model.mixstyle_stages) == (2, 3)
    assert resolve_gradcam_target_layer(model) is model.gradcam_layer


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
def test_sonoglore_learnable_stage_fusion_outputs_two_logits() -> None:
    """Verify learnable normalized stage fusion constructs and records active weights."""
    model = create_classifier(
        "sonoglore_convnext_tiny",
        pretrained=False,
        in_chans=3,
        num_classes=2,
        active_stage_indices=[2, 3, 4],
        stage_proj_dims=[192, 256, 320],
        stage_fusion_weights=[0.8, 1.0, 1.0],
        use_stage4_attention=False,
        use_projection_grn=True,
        use_projection_eca=True,
        learnable_stage_fusion=True,
    )
    batch = classifier.torch.zeros((2, 3, 224, 224), dtype=classifier.torch.float32)

    logits = model(batch)

    assert tuple(logits.shape) == (2, 2)
    assert model.learnable_stage_fusion is True
    assert model.stage_fusion_logits is not None
    assert model.last_stage_fusion_weights is not None
    assert tuple(model.last_stage_fusion_weights.shape) == (3,)
    assert classifier.torch.isclose(
        model.last_stage_fusion_weights.sum(),
        classifier.torch.tensor(3.0, dtype=model.last_stage_fusion_weights.dtype),
        atol=1e-5,
    )
    assert resolve_gradcam_target_layer(model) is model.gradcam_layer


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
def test_sonoglore_selective_local_mixer_outputs_two_logits() -> None:
    """Verify the selective receptive-field local mixer constructs and runs."""
    model = create_classifier(
        "sonoglore_convnext_tiny",
        pretrained=False,
        in_chans=3,
        num_classes=2,
        active_stage_indices=[2, 3, 4],
        stage_proj_dims=[192, 256, 320],
        stage_fusion_weights=[0.8, 1.0, 1.0],
        use_stage4_attention=False,
        use_projection_grn=True,
        use_projection_eca=True,
        use_stage_local_mixer=True,
        stage_local_mixer_stages=[2, 3],
        stage_local_mixer_square_kernel=3,
        stage_local_mixer_band_kernel_sizes=[7, 11],
    )
    batch = classifier.torch.zeros((2, 3, 224, 224), dtype=classifier.torch.float32)

    logits = model(batch)

    assert tuple(logits.shape) == (2, 2)
    assert model.use_stage_local_mixer is True
    assert tuple(model.stage_local_mixer_stages) == (2, 3)
    assert set(model.stage_local_mixers.keys()) == {"2", "3"}
    assert resolve_gradcam_target_layer(model) is model.gradcam_layer


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
def test_sonoglore_projection_spectral_gate_outputs_two_logits() -> None:
    """Verify frequency-aware projection gating constructs and records per-stage weights."""
    model = create_classifier(
        "sonoglore_convnext_tiny",
        pretrained=False,
        in_chans=3,
        num_classes=2,
        active_stage_indices=[2, 3, 4],
        stage_proj_dims=[192, 256, 320],
        stage_fusion_weights=[0.8, 1.0, 1.0],
        use_stage4_attention=False,
        use_projection_grn=True,
        use_projection_eca=True,
        use_projection_spectral_gate=True,
        projection_spectral_gate_stages=[2, 4],
        projection_spectral_gate_freq_size=4,
    )
    batch = classifier.torch.zeros((2, 3, 224, 224), dtype=classifier.torch.float32)

    logits = model(batch)

    assert tuple(logits.shape) == (2, 2)
    assert model.use_projection_spectral_gate is True
    assert tuple(model.projection_spectral_gate_stages) == (2, 4)
    assert set(model.projection_spectral_gates.keys()) == {"2", "4"}
    assert model.last_projection_spectral_gate_weights is not None
    assert set(model.last_projection_spectral_gate_weights.keys()) == {"2", "4"}
    assert tuple(model.last_projection_spectral_gate_weights["2"].shape) == (2, 192)
    assert tuple(model.last_projection_spectral_gate_weights["4"].shape) == (2, 320)
    assert resolve_gradcam_target_layer(model) is model.gradcam_layer


@pytest.mark.skipif(classifier.timm is None, reason="timm is not installed")
def test_load_classifier_restores_sonoglore_checkpoint(tmp_path) -> None:
    """Verify SonoGloReNet checkpoints reload through the shared classifier loader."""
    model = create_classifier(
        "sonoglore_convnext_tiny",
        pretrained=False,
        in_chans=3,
        num_classes=2,
    )
    checkpoint_path = tmp_path / "sonoglore.pt"
    classifier.torch.save({"state_dict": model.state_dict()}, checkpoint_path)

    restored = load_classifier(
        {
            "name": "sonoglore_convnext_tiny",
            "pretrained": False,
            "in_chans": 3,
            "num_classes": 2,
        },
        checkpoint_path=checkpoint_path,
    )

    assert restored is not None

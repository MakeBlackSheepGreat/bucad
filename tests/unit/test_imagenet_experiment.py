"""Unit tests for ImageNet preparation and evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments import imagenet


def test_archive_status_reports_missing_archive(tmp_path: Path) -> None:
    """Verify missing ImageNet archives are reported without crashing."""
    status = imagenet._archive_status(
        tmp_path,
        "ILSVRC2012_img_val.tar",
        "29b22e2961454d5413ddabcf34fc5622",
        skip_md5=False,
    )

    assert status["exists"] is False
    assert status["md5_ok"] is False
    assert status["actual_md5"] is None


def test_archive_status_can_skip_md5_for_present_archive(tmp_path: Path) -> None:
    """Verify skip-md5 mode accepts an existing archive placeholder."""
    archive = tmp_path / "ILSVRC2012_img_val.tar"
    archive.write_bytes(b"placeholder")

    status = imagenet._archive_status(
        tmp_path,
        archive.name,
        "29b22e2961454d5413ddabcf34fc5622",
        skip_md5=True,
    )

    assert status["exists"] is True
    assert status["md5_ok"] is True
    assert status["size_bytes"] == len(b"placeholder")


def test_ablation_markdown_renders_core_metrics() -> None:
    """Verify ImageNet ablation Markdown contains Top-1/Top-5 fields."""
    lines = imagenet._ablation_markdown(
        {
            "root": "datasets/imagenet",
            "split": "val",
            "dataset_size": 50000,
            "max_batches": 1,
            "rows": [
                {
                    "id": "v1",
                    "name": "ConvNeXt-Tiny",
                    "metrics": {
                        "top1": 0.81,
                        "top5": 0.95,
                        "images_per_second": 123.4,
                    },
                    "parameter_count": 123456,
                }
            ],
        }
    )

    text = "\n".join(lines)
    assert "Top-1" in text
    assert "ConvNeXt-Tiny" in text
    assert "0.8100" in text


def test_ablation_markdown_renders_requires_training_rows() -> None:
    """Verify missing trained checkpoints are visible in ablation Markdown."""
    lines = imagenet._ablation_markdown(
        {
            "root": "datasets/imagenet",
            "split": "val",
            "dataset_size": None,
            "max_batches": None,
            "rows": [
                {
                    "id": "v1",
                    "name": "ConvNeXt-Tiny",
                    "status": "requires_training",
                    "metrics": None,
                    "parameter_count": None,
                }
            ],
        }
    )

    assert "requires training" in "\n".join(lines)


def test_resolve_data_config_treats_null_overrides_as_model_defaults() -> None:
    """Verify null preprocess values preserve timm/model defaults."""
    resolved = imagenet._resolve_data_config(
        object(),
        {
            "image_size": 224,
            "preprocess": {
                "use_timm_data_config": False,
                "mean": None,
                "std": None,
                "interpolation": None,
                "crop_pct": None,
            },
        },
    )

    assert resolved["mean"] == (0.485, 0.456, 0.406)
    assert resolved["std"] == (0.229, 0.224, 0.225)
    assert resolved["crop_pct"] == 0.875


def test_require_prepared_imagenet_reports_missing_archives(tmp_path: Path) -> None:
    """Verify eval preflight reports actionable missing archive names."""
    with pytest.raises(FileNotFoundError) as exc_info:
        imagenet._require_prepared_imagenet(tmp_path, "val")

    message = str(exc_info.value)
    assert "ILSVRC2012_devkit_t12.tar.gz" in message
    assert "ILSVRC2012_img_val.tar" in message
    assert "prepare_imagenet.py" in message


def test_build_model_supports_v1_imagenet_head() -> None:
    """Verify the V1 alias can be constructed with a 1000-class ImageNet head."""
    if imagenet.torch is None:
        return
    model = imagenet._build_model(
        {
            "name": "ConvNeXt-Tiny",
            "pretrained": False,
            "in_chans": 3,
            "num_classes": 1000,
        },
        device="cpu",
    )

    assert imagenet._count_parameters(model) > 0


def test_scaled_learning_rate_matches_convnext_recipe() -> None:
    """Verify ConvNeXt base LR scales linearly from the reference batch size."""
    lr = imagenet._scaled_learning_rate(
        {
            "base_learning_rate": 0.004,
            "reference_batch_size": 4096,
            "scale_lr_by_batch_size": True,
        },
        batch_size=64,
    )

    assert lr == pytest.approx(0.004 * 64 / 4096)


def test_resolve_checkpoint_path_uses_project_relative_paths(tmp_path: Path) -> None:
    """Verify configured checkpoints resolve relative to the project root."""
    class Paths:
        """Minimal project-path container for checkpoint resolution."""

        project_root = tmp_path

    resolved = imagenet._resolve_checkpoint_path(
        {"model": {"checkpoint": "./artifacts/checkpoints/model.pt"}},
        Paths(),
    )

    assert resolved == (tmp_path / "artifacts" / "checkpoints" / "model.pt").resolve()


def test_v1_fair_eval_requires_full_checkpoint(tmp_path: Path, monkeypatch) -> None:
    """Verify V1 eval refuses backbone-only pretrained evaluation."""
    if imagenet.torch is None:
        return
    class Paths:
        """Minimal project-path container for ImageNet evaluation."""

        project_root = tmp_path
        imagenet_root = tmp_path / "imagenet"
        reports_root = tmp_path / "reports"

    monkeypatch.setattr(
        imagenet,
        "load_project_config",
        lambda _path: (
            {
                "seed": 42,
                "device": "cpu",
                "dataset": {"root": str(tmp_path / "imagenet"), "split": "val"},
                "model": {
                    "name": "ConvNeXt-Tiny",
                    "pretrained": True,
                    "in_chans": 3,
                    "num_classes": 1000,
                },
            },
            Paths(),
        ),
    )
    monkeypatch.setattr(imagenet, "_require_prepared_imagenet", lambda _root, _split: None)

    with pytest.raises(ValueError, match="requires a full ImageNet-trained checkpoint"):
        imagenet.evaluate_imagenet_config("dummy.yml")


def test_official_pretrained_baseline_requires_explicit_opt_in() -> None:
    """Verify official pretrained evaluation needs an explicit configuration flag."""
    model_cfg = {"name": "convnext_tiny", "pretrained": True}

    assert imagenet._requires_trained_checkpoint(model_cfg) is True
    assert imagenet._requires_trained_checkpoint(
        model_cfg,
        {"allow_official_pretrained_baseline": True},
    ) is False


def test_run_ablation_marks_missing_checkpoints_requires_training(tmp_path: Path, monkeypatch) -> None:
    """Verify missing ablation checkpoints are reported instead of crashing."""
    if imagenet.torch is None:
        return
    class Paths:
        """Minimal project-path container for ablation reports."""

        project_root = tmp_path
        imagenet_root = tmp_path / "imagenet"
        reports_root = tmp_path / "reports"

    monkeypatch.setattr(
        imagenet,
        "load_project_config",
        lambda _path: (
            {
                "seed": 42,
                "device": "cpu",
                "dataset": {"root": str(tmp_path / "imagenet"), "split": "val"},
                "models": [
                    {
                        "id": "v1",
                        "name": "ConvNeXt-Tiny",
                        "model": {
                            "name": "ConvNeXt-Tiny",
                            "pretrained": False,
                            "in_chans": 3,
                            "num_classes": 1000,
                            "checkpoint": "./artifacts/checkpoints/missing.pt",
                        },
                    }
                ],
                "output": {
                    "report_name": "ablation.json",
                    "markdown_name": "ablation.md",
                },
            },
            Paths(),
        ),
    )
    monkeypatch.setattr(imagenet, "_require_prepared_imagenet", lambda _root, _split: None)

    report = imagenet.run_imagenet_ablation("dummy.yml")

    assert report["rows"][0]["status"] == "requires_training"
    assert report["dataset_size"] is None
    assert (tmp_path / "reports" / "ablation.json").exists()


def test_train_imagenet_smoke_writes_checkpoint_and_report(tmp_path: Path, monkeypatch) -> None:
    """Verify the ImageNet training loop can run one synthetic batch."""
    if imagenet.torch is None:
        return
    torch = imagenet.torch

    class Paths:
        """Minimal project-path container for the synthetic training run."""

        project_root = tmp_path
        imagenet_root = tmp_path / "imagenet"
        checkpoints_root = tmp_path / "checkpoints"
        reports_root = tmp_path / "reports"

    class TinyImageNet(torch.utils.data.Dataset):
        """Four-sample synthetic ImageNet-like dataset for the smoke test."""

        def __init__(self, split: str) -> None:
            """Record the requested dataset split."""
            self.split = split

        def __len__(self) -> int:
            """Return the fixed smoke-test sample count."""
            return 4

        def __getitem__(self, index: int):
            """Return a random tiny image and deterministic binary label."""
            image = torch.rand(3, 8, 8)
            label = index % 2
            return image, label

    def fake_loader(*, config, model, root, split, is_training=False):
        """Build the deterministic synthetic data loader used by this test."""
        return torch.utils.data.DataLoader(
            TinyImageNet(split),
            batch_size=2,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

    monkeypatch.setattr(
        imagenet,
        "load_project_config",
        lambda _path: (
            {
                "seed": 42,
                "device": "cpu",
                "dataset": {"root": str(tmp_path / "imagenet"), "num_classes": 2},
                "model": {
                    "name": "tiny_cnn",
                    "pretrained": False,
                    "in_chans": 3,
                    "num_classes": 2,
                },
                "data": {
                    "image_size": 8,
                    "batch_size": 2,
                    "num_workers": 0,
                    "preprocess": {"use_timm_data_config": False},
                },
                "training": {
                    "epochs": 1,
                    "base_learning_rate": 0.001,
                    "reference_batch_size": 2,
                    "scale_lr_by_batch_size": False,
                    "mixup_alpha": 0.0,
                    "cutmix_alpha": 0.0,
                    "label_smoothing": 0.0,
                    "amp": False,
                },
                "evaluation": {"amp": False},
                "output": {
                    "checkpoint_name": "tiny_imagenet.pt",
                    "report_name": "tiny_imagenet.json",
                },
            },
            Paths(),
        ),
    )
    monkeypatch.setattr(imagenet, "_require_prepared_imagenet", lambda _root, _split: None)
    monkeypatch.setattr(imagenet, "_build_imagenet_loader", fake_loader)

    report = imagenet.train_imagenet_config(
        "dummy.yml",
        max_batches=1,
        max_val_batches=1,
    )

    assert Path(report["checkpoint_path"]).exists()
    assert Path(report["report_path"]).exists()
    assert report["epoch_reports"][0]["train"]["samples"] == 2


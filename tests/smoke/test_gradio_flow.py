from __future__ import annotations

import pytest


def test_gradio_app_builds_when_dependency_is_installed() -> None:
    pytest.importorskip("gradio")
    from app.main import build_app

    app = build_app()
    assert app is not None

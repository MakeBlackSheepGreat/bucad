"""Run environment checks and the supported BUCAD unit/smoke test suites."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def run(command: list[str]) -> int:
    """Run one validation command and return its process exit code."""
    completed = subprocess.run(command, check=False)
    return completed.returncode


def main() -> int:
    """Run environment reporting followed by the supported unit and smoke tests."""
    root = Path(__file__).resolve().parent
    env_code = run([sys.executable, str(root / "check_env.py")])
    pytest_tmp = Path(tempfile.mkdtemp(prefix="pytest-bucad-", dir=root / "tmp"))
    test_code = run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit",
            "tests/smoke",
            "--basetemp",
            str(pytest_tmp),
        ]
    )
    return 0 if env_code == 0 and test_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

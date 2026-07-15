from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run_script(script_name: str) -> subprocess.CompletedProcess[str]:
    python_executable = Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "python.exe"
    return subprocess.run(
        [str(python_executable), str(SCRIPT_DIR / script_name)],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        check=False,
    )


def test_generate_reconstruction_script_creates_output(tmp_path: Path) -> None:
    result = run_script("generate_reconstruction.py")
    assert result.returncode == 0, result.stderr
    assert (RESULTS_DIR / "reconstruction.png").exists()


def test_generate_kl_plot_script_creates_output() -> None:
    result = run_script("generate_kl_plot.py")
    assert result.returncode == 0, result.stderr
    assert (RESULTS_DIR / "kl_plot.png").exists()


def test_export_samples_script_creates_output() -> None:
    result = run_script("export_samples.py")
    assert result.returncode == 0, result.stderr
    assert (RESULTS_DIR / "generated_samples.png").exists()

"""Thor session preparation: nvpmodel + jetson_clocks when available."""

from __future__ import annotations

import shutil
import subprocess
import sys


def prepare_thor_session(*, nvpmodel_mode: str = "0") -> list[str]:
    """Best-effort lock power mode and clocks. Returns notes about what ran or was skipped."""
    notes: list[str] = []
    nvpmodel = shutil.which("nvpmodel")
    if nvpmodel is None:
        notes.append("nvpmodel not found; skipped")
    else:
        completed = _run_privileged([nvpmodel, "-m", nvpmodel_mode])
        if completed.returncode != 0:
            notes.append(f"nvpmodel failed: {_err(completed)}")
        else:
            notes.append(f"nvpmodel -m {nvpmodel_mode}")

    jetson_clocks = shutil.which("jetson_clocks")
    if jetson_clocks is None:
        notes.append("jetson_clocks not found; skipped")
    else:
        completed = _run_privileged([jetson_clocks])
        if completed.returncode != 0:
            notes.append(f"jetson_clocks failed: {_err(completed)}")
        else:
            notes.append("jetson_clocks")

    for note in notes:
        print(note, file=sys.stderr)
    return notes


def _run_privileged(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Prefer passwordless sudo (Thor eval account); fall back to direct exec."""
    sudo = shutil.which("sudo")
    if sudo is not None:
        via_sudo = subprocess.run([sudo, "-n", *cmd], capture_output=True, text=True)
        if via_sudo.returncode == 0 or "password" not in (via_sudo.stderr or "").lower():
            return via_sudo
    return subprocess.run(cmd, capture_output=True, text=True)


def _err(completed: subprocess.CompletedProcess[str]) -> str:
    return (completed.stderr or completed.stdout or "").strip()

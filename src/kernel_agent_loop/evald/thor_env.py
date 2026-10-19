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
        completed = subprocess.run(
            [nvpmodel, "-m", nvpmodel_mode],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            notes.append(f"nvpmodel failed: {completed.stderr.strip() or completed.stdout.strip()}")
        else:
            notes.append(f"nvpmodel -m {nvpmodel_mode}")

    jetson_clocks = shutil.which("jetson_clocks")
    if jetson_clocks is None:
        notes.append("jetson_clocks not found; skipped")
    else:
        completed = subprocess.run([jetson_clocks], capture_output=True, text=True)
        if completed.returncode != 0:
            notes.append(f"jetson_clocks failed: {completed.stderr.strip() or completed.stdout.strip()}")
        else:
            notes.append("jetson_clocks")

    for note in notes:
        print(note, file=sys.stderr)
    return notes

"""Runs inside the ncu container: launch a cubin once per shape from a job dir; no agent code executes here.

The job dir holds kernel.cubin, request.json (shape label, dims, already-evaluated launch config, arrays
file) and one .npz of inputs ("in:<name>") and outputs ("out:<name>") per shape.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from kernel_agent_loop.backends.cuda import CudaKernel, checked_launch_config
from kernel_agent_loop.core.types import Shape


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    job = Path(argv[0])
    request = json.loads((job / "request.json").read_text())
    configs = {item["label"]: checked_launch_config(item["config"]) for item in request["shapes"]}
    kernel = CudaKernel((job / "kernel.cubin").read_bytes(), lambda shape: configs[shape.label])
    try:
        for item in request["shapes"]:
            with np.load(job / item["arrays"]) as arrays:
                inputs = {k[3:]: arrays[k] for k in arrays.files if k.startswith("in:")}
                outputs = {k[4:]: arrays[k].copy() for k in arrays.files if k.startswith("out:")}
            kernel(inputs, outputs, Shape(item["label"], item["dims"]))
            print(f"launched {item['label']}", file=sys.stderr)
    finally:
        kernel.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

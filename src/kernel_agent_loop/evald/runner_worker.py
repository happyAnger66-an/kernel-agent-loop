"""Child process entry: load artifact, serve run() over stdout JSONL; logs go to stderr."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from kernel_agent_loop.core.types import Shape
from kernel_agent_loop.evald.array_codec import decode_array, encode_array


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: runner_worker ARTIFACT_DIR [--incumbent DIR] [--fake]", file=sys.stderr)
        return 2
    artifact_dir = Path(argv[0])
    fake = "--fake" in argv[1:]
    incumbent_dir = Path(argv[argv.index("--incumbent") + 1]) if "--incumbent" in argv[1:] else None

    # stdout is reserved for JSONL; divert fd 1 library noise to stderr (ExecBench pattern).
    real_stdout = os.fdopen(os.dup(1), "w", encoding="utf-8", closefd=True)
    os.dup2(2, 1)

    try:
        run = _load_run(artifact_dir, fake=fake)
    except Exception as exc:
        real_stdout.write(json.dumps({"ok": False, "error": f"load failed: {exc}"}, allow_nan=False) + "\n")
        real_stdout.flush()
        return 1
    incumbent = None
    if incumbent_dir is not None:
        try:
            incumbent = _load_run(incumbent_dir, fake=fake)
        except Exception as exc:
            print(f"incumbent not loaded: {exc}", file=sys.stderr)
    real_stdout.write(json.dumps({"ok": True, "incumbent": incumbent is not None}) + "\n")
    real_stdout.flush()

    staged: dict = {}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        if request.get("cmd") == "quit":
            return 0
        try:
            response = _handle_run(run, incumbent, request, staged)
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        real_stdout.write(json.dumps(response, allow_nan=False) + "\n")
        real_stdout.flush()
    return 0


def _load_run(artifact_dir: Path, *, fake: bool):
    if fake:
        return _load_fake(artifact_dir)
    from kernel_agent_loop.backends.cuda import Artifact, CudaBackend

    cubin = artifact_dir / "kernel.cubin"
    if not cubin.is_file():
        # FakeBackend-style artifacts may point cubin_path at kernel.cu; real CUDA needs .cubin.
        raise FileNotFoundError(f"missing {cubin}")
    backend = CudaBackend()
    return backend.load(Artifact(cubin_path=cubin, source_dir=artifact_dir))


def _load_fake(artifact_dir: Path):
    """Test-only worker: interpret kernel.cu markers like FakeBackend in unit tests."""
    import numpy as np

    text = (artifact_dir / "kernel.cu").read_text()
    if "COMPILE_FAIL" in text or "CANDIDATE" not in text:
        raise RuntimeError("error: not a candidate")

    def run(inputs, outputs, shape) -> None:
        if "SLOW" in text:
            import time

            time.sleep(0.02)
        if "WRONG" in text:
            for array in outputs.values():
                array.fill(0)
            return
        # Minimal RMSNorm-shaped reference for public shapes used in tests.
        x = inputs["x"].astype(np.float64)
        weight = inputs["weight"].astype(np.float64)
        y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) * weight
        outputs["y"][...] = y.astype(inputs["x"].dtype)

    return run


def _handle_run(run, incumbent, request: dict, staged: dict) -> dict:
    """`stage` keeps arrays for later bench calls; `bench`/`bench_pair` use them; `run` brings its own."""
    shape = Shape(label=request["shape"]["label"], dims=request["shape"]["dims"])
    cmd = request.get("cmd")
    if cmd == "launch_config":
        return {"ok": True, "config": run.launch_config(shape)}
    if cmd in ("bench", "bench_pair"):
        if staged.get("shape") != shape:
            raise RuntimeError(f"{cmd} before stage for {shape.label}")
        if cmd == "bench":
            return {"ok": True, "us": _bench(run, staged["inputs"], staged["outputs"], shape)}
        if incumbent is None:
            raise RuntimeError("bench_pair without an incumbent")
        cand_us, inc_us = _bench_pair(run, incumbent, staged["inputs"], staged["outputs"], shape)
        return {"ok": True, "candidate_us": cand_us, "incumbent_us": inc_us}
    inputs = {name: decode_array(payload) for name, payload in request["inputs"].items()}
    outputs = {name: decode_array(payload) for name, payload in request["outputs"].items()}
    if cmd == "stage":
        staged.update(shape=shape, inputs=inputs, outputs=outputs)
        return {"ok": True}
    run(inputs, outputs, shape)
    return {"ok": True, "outputs": {name: encode_array(value) for name, value in outputs.items()}}


def _bench_pair(run, incumbent, inputs: dict, outputs: dict, shape: Shape) -> tuple[float, float]:
    pair = getattr(run, "bench_pair_with", None)
    if callable(pair):
        return pair(incumbent, inputs, outputs, shape)
    return _bench(run, inputs, outputs, shape), _bench(incumbent, inputs, outputs, shape)


def _bench(run, inputs: dict, outputs: dict, shape: Shape) -> float:
    bench = getattr(run, "bench", None)
    if callable(bench):
        return float(bench(inputs, outputs, shape))
    started = time.perf_counter()
    run(inputs, outputs, shape)
    return (time.perf_counter() - started) * 1e6


if __name__ == "__main__":
    raise SystemExit(main())

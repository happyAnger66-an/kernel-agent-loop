"""Measured per-shape hardware floor: a touch kernel that reads every input byte and writes every output byte.

The floor includes launch overhead, so current_us / floor_us is the headroom an impl has left on the
same bandwidth path. It is not a proof of the minimum: a kernel that skips or re-uses bytes can beat it,
which is why the gate treats a candidate far below its floor as suspicious.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from kernel_agent_loop.core.types import Shape

_VEC_BYTES = 16
_BLOCK = 256

KERNEL_SRC = r"""
extern "C" __global__ void touch(const uint4* __restrict__ src, uint4* __restrict__ dst, int n_in, int n_out) {
    uint4 acc = make_uint4(0, 0, 0, 0);
    int n = n_in > n_out ? n_in : n_out;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        uint4 v = make_uint4(0, 0, 0, 0);
        if (i < n_in) v = src[i];
        if (i < n_out) {
            dst[i] = v;
        } else {
            acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
        }
    }
    // Reads past n_out must stay live; this store practically never fires.
    if (n_out > 0 && (acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9u) dst[0].x ^= acc.x;
}
"""


def launch_config(shape: Shape) -> dict:
    n = max(int(shape.dims["n_in"]), int(shape.dims["n_out"]), 1)
    return {
        "kernel": "touch",
        "grid": ((n + _BLOCK - 1) // _BLOCK, 1, 1),
        "block": (_BLOCK, 1, 1),
        "smem": 0,
        "args": ["src", "dst", "n_in", "n_out"],
    }


def shape_bytes(spec, shape: Shape) -> tuple[int, int]:
    """(bytes read, bytes written) by an ideal impl: every input once, every output once."""
    inputs = spec.make_inputs(shape, seed=0)
    read = sum(int(np.asarray(array).nbytes) for array in inputs.values())
    written = sum(
        int(np.prod(out_shape)) * np.dtype(dtype).itemsize
        for out_shape, dtype in spec.output_specs(shape).values()
    )
    return read, written


class FloorProbe:
    """Builds the touch kernel once; a floor is a low percentile over `rounds` jittered benches.

    On Thor the touch kernel's time swings about +-15% with buffer placement, and the median of one
    process differs from the next by 5-10%. A low percentile tracks the well-placed case, which is what
    an impl can reach, and moves only about +-5% between processes.
    """

    def __init__(self, *, rounds: int = 30, percentile: float = 10.0) -> None:
        self.rounds = rounds
        self.percentile = percentile
        self._kernel = None

    def floor_us(self, read_bytes: int, written_bytes: int) -> float:
        n_in = -(-read_bytes // _VEC_BYTES)
        n_out = -(-written_bytes // _VEC_BYTES)
        shape = Shape("floor", {"n_in": n_in, "n_out": n_out})
        inputs = {"src": np.zeros(max(n_in, 1) * _VEC_BYTES, dtype=np.uint8)}
        outputs = {"dst": np.zeros(max(n_out, 1) * _VEC_BYTES, dtype=np.uint8)}
        kernel = self._load()
        samples = [kernel.bench(inputs, outputs, shape) for _ in range(self.rounds)]
        return float(np.percentile(samples, self.percentile))

    def __call__(self, spec, shape: Shape) -> float:
        return self.floor_us(*shape_bytes(spec, shape))

    def close(self) -> None:
        if self._kernel is not None:
            self._kernel.close()
            self._kernel = None

    def _load(self):
        if self._kernel is None:
            from kernel_agent_loop.backends.cuda import CudaBackend, CudaKernel

            source = Path(tempfile.mkdtemp(prefix="kal-floor-"))
            (source / "kernel.cu").write_text(KERNEL_SRC)
            artifact = CudaBackend().build(source)
            self._kernel = CudaKernel(artifact.cubin_path.read_bytes(), launch_config)
        return self._kernel


def main(argv: list[str] | None = None) -> int:
    from kernel_agent_loop.core.opspec import load_opspec
    from kernel_agent_loop.evald.config import EvaldConfig

    parser = argparse.ArgumentParser(description="Print measured hardware floors for an op's shapes")
    parser.add_argument("--config", required=True, type=Path, help="EvaldConfig JSON path")
    parser.add_argument("--op", default="rmsnorm")
    args = parser.parse_args(argv)

    config = EvaldConfig.load(args.config)
    ops_root = config.ops_root or Path(config.trusted) / "ops"
    spec = load_opspec(ops_root, args.op)
    probe = FloorProbe()
    try:
        print(f"launch floor (0 bytes): {probe.floor_us(0, 0):.2f} us")
        big = 256 << 20
        print(f"copy 256 MiB: {2 * big / probe.floor_us(big, big) / 1e3:.1f} GB/s")
        for shape in list(spec.public_shapes()) + list(config.hidden_shapes):
            read, written = shape_bytes(spec, shape)
            us = probe.floor_us(read, written)
            print(f"{shape.label:>16}: {us:8.2f} us  ({read + written} B, {(read + written) / us / 1e3:.1f} GB/s)")
    finally:
        probe.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

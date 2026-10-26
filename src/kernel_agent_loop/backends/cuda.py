import importlib.util
import random
import sys
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class BuildError(Exception):
    pass


@dataclass(frozen=True)
class Artifact:
    cubin_path: Path
    source_dir: Path


class CudaBackend:
    name = "cuda"

    def build(self, source_dir: Path) -> Artifact:
        source_dir = Path(source_dir)
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            raise BuildError("nvcc not found")
        out_dir = Path(tempfile.mkdtemp(prefix="kal-cubin-"))
        cubin = out_dir / "kernel.cubin"
        compiled = subprocess.run(
            [nvcc, "-cubin", f"-arch={_gpu_arch()}", str(source_dir / "kernel.cu"), "-o", str(cubin)],
            capture_output=True,
            text=True,
        )
        if compiled.returncode != 0:
            raise BuildError(compiled.stderr or compiled.stdout)
        return Artifact(cubin, source_dir)

    def load(self, artifact: Artifact) -> "CudaKernel":
        launch_config = _load_launch_config(artifact.source_dir / "launch.py")
        return CudaKernel(artifact.cubin_path.read_bytes(), launch_config)


class CudaKernel:
    """A cubin loaded once into the process-wide primary context.

    Calling it runs one launch with host<->device copies (for correctness checks).
    bench() times launches alone with CUDA events, flushing L2 before each one.
    """

    def __init__(self, cubin: bytes, launch_config) -> None:
        from cuda.bindings import driver as cuda

        self._launch_config = launch_config
        _make_current()
        self._module = _check(cuda.cuModuleLoadData(cubin))
        self._functions: dict[str, object] = {}

    def __call__(self, inputs: dict, outputs: dict, shape) -> None:
        from cuda.bindings import driver as cuda

        _make_current()
        config = self._launch_config(shape)
        with _DeviceBuffers(inputs, outputs) as buffers:
            self._launch_once(config, buffers.launch_args(config, shape))
            _check(cuda.cuCtxSynchronize())
            buffers.copy_outputs_back()

    def launch_config(self, shape) -> dict:
        """Launch config for shape as plain JSON types; raises BuildError if it is malformed."""
        return checked_launch_config(self._launch_config(shape))

    def bench(self, inputs: dict, outputs: dict, shape, *, iters: int = 20, warmup: int = 3) -> float:
        """Median per-launch device time in microseconds."""
        return bench_kernels([self], inputs, outputs, shape, iters=iters, warmup=warmup)[0]

    def bench_pair_with(self, other: "CudaKernel", inputs: dict, outputs: dict, shape) -> tuple[float, float]:
        """(self, other) median launch times on the same buffers, launches alternating."""
        first, second = bench_kernels([self, other], inputs, outputs, shape)
        return first, second

    def close(self) -> None:
        from cuda.bindings import driver as cuda

        _make_current()
        if self._module is not None:
            cuda.cuModuleUnload(self._module)
            self._module = None

    def _launch_once(self, config: dict, arg_ptrs) -> None:
        from cuda.bindings import driver as cuda

        grid, block = config["grid"], config["block"]
        _check(
            cuda.cuLaunchKernel(
                self._function(config["kernel"]),
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                int(config["smem"]),
                0,
                arg_ptrs,
                0,
            )
        )

    def _function(self, name: str):
        from cuda.bindings import driver as cuda

        if name not in self._functions:
            self._functions[name] = _check(cuda.cuModuleGetFunction(self._module, name.encode()))
        return self._functions[name]


def bench_kernels(kernels: list[CudaKernel], inputs: dict, outputs: dict, shape, *,
                  iters: int = 20, warmup: int = 3) -> list[float]:
    """Median per-launch microseconds for each kernel, all launched on one set of buffers.

    Buffers sit at a fresh random offset on every call. Launch order rotates each iteration and
    L2 is flushed before every launch, so placement and cache state hit every kernel alike.
    """
    from cuda.bindings import driver as cuda

    _make_current()
    configs = [kernel._launch_config(shape) for kernel in kernels]
    flush_ptr, flush_bytes = _flush_buffer()
    with _DeviceBuffers(inputs, outputs, jitter=True) as buffers:
        args = [buffers.launch_args(config, shape) for config in configs]
        for _ in range(warmup):
            for kernel, config, arg_ptrs in zip(kernels, configs, args):
                kernel._launch_once(config, arg_ptrs)
        events = [[(_check(cuda.cuEventCreate(0)), _check(cuda.cuEventCreate(0))) for _ in range(iters)]
                  for _ in kernels]
        try:
            for i in range(iters):
                for step in range(len(kernels)):
                    k = (i + step) % len(kernels)
                    start, end = events[k][i]
                    _check(cuda.cuMemsetD8Async(flush_ptr, i & 0xFF, flush_bytes, 0))
                    _check(cuda.cuEventRecord(start, 0))
                    kernels[k]._launch_once(configs[k], args[k])
                    _check(cuda.cuEventRecord(end, 0))
            _check(cuda.cuCtxSynchronize())
            return [
                float(np.median([_check(cuda.cuEventElapsedTime(s, e)) * 1000.0 for s, e in pairs]))
                for pairs in events
            ]
        finally:
            for pairs in events:
                for s, e in pairs:
                    cuda.cuEventDestroy(s)
                    cuda.cuEventDestroy(e)


class _DeviceBuffers:
    """Device copies of every input and output array; freed on exit.

    With jitter, each buffer starts at a random 256-byte-aligned offset inside a padded allocation.
    Kernel time on Thor depends on buffer placement, and cuMemAlloc hands back the same addresses
    within a process, so without jitter every sample in a process sees the same placement.
    """

    _JITTER_BYTES = 8 << 20
    _ALIGN = 256

    def __init__(self, inputs: dict, outputs: dict, *, jitter: bool = False) -> None:
        self._inputs = inputs
        self._outputs = outputs
        self._jitter = jitter
        self._allocations: list = []
        self._device: dict[str, int] = {}
        self._hosts: dict[str, np.ndarray] = {}
        self._keepalive: list = []

    def __enter__(self) -> "_DeviceBuffers":
        from cuda.bindings import driver as cuda

        try:
            for name, source in {**self._inputs, **self._outputs}.items():
                host = np.ascontiguousarray(source)
                pad = self._JITTER_BYTES if self._jitter else 0
                base = _check(cuda.cuMemAlloc(host.nbytes + pad))
                self._allocations.append(base)
                ptr = int(base) + (random.randrange(0, pad, self._ALIGN) if pad else 0)
                _check(cuda.cuMemcpyHtoD(ptr, host.ctypes.data, host.nbytes))
                self._device[name] = ptr
                self._hosts[name] = host
        except BaseException:
            self._free()
            raise
        return self

    def launch_args(self, config: dict, shape):
        import ctypes

        values = []
        for name in config["args"]:
            if name in self._device:
                values.append(ctypes.c_uint64(self._device[name]))
            elif name in shape.dims:
                values.append(ctypes.c_int(int(shape.dims[name])))
            else:
                raise BuildError(f"launch arg {name} is not an input, output, or shape dim")
        self._keepalive.append(values)
        return (ctypes.c_void_p * len(values))(*(ctypes.addressof(v) for v in values))

    def copy_outputs_back(self) -> None:
        from cuda.bindings import driver as cuda

        for name in self._outputs:
            host = self._hosts[name]
            _check(cuda.cuMemcpyDtoH(host.ctypes.data, self._device[name], host.nbytes))
            self._outputs[name][...] = host

    def __exit__(self, *_) -> None:
        self._free()

    def _free(self) -> None:
        from cuda.bindings import driver as cuda

        for ptr in self._allocations:
            cuda.cuMemFree(ptr)
        self._allocations = []


def _flush_buffer() -> tuple[object, int]:
    from cuda.bindings import driver as cuda

    if "flush" not in _PRIMARY:
        _make_current()
        l2 = _check(cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, _device()))
        size = max(2 * int(l2), 8 << 20)
        _PRIMARY["flush"] = (_check(cuda.cuMemAlloc(size)), size)
    return _PRIMARY["flush"]


_PRIMARY: dict = {}


def _device():
    from cuda.bindings import driver as cuda

    if "device" not in _PRIMARY:
        _check(cuda.cuInit(0))
        _PRIMARY["device"] = _check(cuda.cuDeviceGet(0))
    return _PRIMARY["device"]


def _make_current() -> None:
    from cuda.bindings import driver as cuda

    if "context" not in _PRIMARY:
        _PRIMARY["context"] = _check(cuda.cuDevicePrimaryCtxRetain(_device()))
    _check(cuda.cuCtxSetCurrent(_PRIMARY["context"]))


def _gpu_arch() -> str:
    from cuda.bindings import driver as cuda

    device = _device()
    major = _check(
        cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device)
    )
    minor = _check(
        cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device)
    )
    return f"sm_{major}{minor}"


def checked_launch_config(config) -> dict:
    try:
        checked = {
            "kernel": str(config["kernel"]),
            "grid": [int(v) for v in config["grid"]],
            "block": [int(v) for v in config["block"]],
            "smem": int(config["smem"]),
            "args": [str(v) for v in config["args"]],
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise BuildError(f"malformed launch config: {exc}") from exc
    if len(checked["grid"]) != 3 or len(checked["block"]) != 3:
        raise BuildError("launch config grid and block need 3 dims")
    if not checked["kernel"].isidentifier():
        raise BuildError(f"launch config kernel {checked['kernel']!r} is not an identifier")
    return checked


def _load_launch_config(path: Path):
    # Unique module name per path so placeholder and real launch.py do not collide in sys.modules.
    mod_name = f"kal_cuda_launch_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise BuildError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "launch_config"):
        raise BuildError(f"{path} has no launch_config")
    return module.launch_config


def _check(result):
    from cuda.bindings import driver as cuda

    if isinstance(result, tuple):
        err, *rest = result
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise BuildError(str(err))
        return rest[0] if len(rest) == 1 else rest
    return result

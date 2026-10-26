import importlib.util
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
        self._flush: tuple[object, int] | None = None

    def __call__(self, inputs: dict, outputs: dict, shape) -> None:
        from cuda.bindings import driver as cuda

        _make_current()
        config = self._launch_config(shape)
        with _DeviceArgs(config, inputs, outputs, shape) as args:
            self._launch_once(config, args)
            _check(cuda.cuCtxSynchronize())
            args.copy_outputs_back()

    def bench(self, inputs: dict, outputs: dict, shape, *, iters: int = 20, warmup: int = 3) -> float:
        """Median per-launch device time in microseconds."""
        from cuda.bindings import driver as cuda

        _make_current()
        config = self._launch_config(shape)
        flush_ptr, flush_bytes = self._flush_buffer()
        with _DeviceArgs(config, inputs, outputs, shape) as args:
            for _ in range(warmup):
                self._launch_once(config, args)
            starts = [_check(cuda.cuEventCreate(0)) for _ in range(iters)]
            ends = [_check(cuda.cuEventCreate(0)) for _ in range(iters)]
            try:
                for i in range(iters):
                    _check(cuda.cuMemsetD8Async(flush_ptr, i & 0xFF, flush_bytes, 0))
                    _check(cuda.cuEventRecord(starts[i], 0))
                    self._launch_once(config, args)
                    _check(cuda.cuEventRecord(ends[i], 0))
                _check(cuda.cuEventSynchronize(ends[-1]))
                times_us = [_check(cuda.cuEventElapsedTime(s, e)) * 1000.0 for s, e in zip(starts, ends)]
            finally:
                for event in starts + ends:
                    cuda.cuEventDestroy(event)
        return float(np.median(times_us))

    def close(self) -> None:
        from cuda.bindings import driver as cuda

        _make_current()
        if self._flush is not None:
            cuda.cuMemFree(self._flush[0])
            self._flush = None
        if self._module is not None:
            cuda.cuModuleUnload(self._module)
            self._module = None

    def _launch_once(self, config: dict, args: "_DeviceArgs") -> None:
        from cuda.bindings import driver as cuda

        grid, block = config["grid"], config["block"]
        _check(
            cuda.cuLaunchKernel(
                self._function(config["kernel"]),
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                int(config["smem"]),
                0,
                args.arg_ptrs,
                0,
            )
        )

    def _function(self, name: str):
        from cuda.bindings import driver as cuda

        if name not in self._functions:
            self._functions[name] = _check(cuda.cuModuleGetFunction(self._module, name.encode()))
        return self._functions[name]

    def _flush_buffer(self) -> tuple[object, int]:
        from cuda.bindings import driver as cuda

        if self._flush is None:
            l2 = _check(
                cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, _device())
            )
            size = max(2 * int(l2), 8 << 20)
            self._flush = (_check(cuda.cuMemAlloc(size)), size)
        return self._flush


class _DeviceArgs:
    """Device copies of the launch args for one shape; freed on exit."""

    def __init__(self, config: dict, inputs: dict, outputs: dict, shape) -> None:
        self._config = config
        self._inputs = inputs
        self._outputs = outputs
        self._shape = shape
        self._allocations: list = []
        self._output_slots: dict = {}
        self._kernel_args: list = []
        self.arg_ptrs = None

    def __enter__(self) -> "_DeviceArgs":
        import ctypes

        from cuda.bindings import driver as cuda

        try:
            for name in self._config["args"]:
                if name in self._inputs or name in self._outputs:
                    source = self._inputs[name] if name in self._inputs else self._outputs[name]
                    host = np.ascontiguousarray(source)
                    ptr = _check(cuda.cuMemAlloc(host.nbytes))
                    self._allocations.append(ptr)
                    _check(cuda.cuMemcpyHtoD(ptr, host.ctypes.data, host.nbytes))
                    self._kernel_args.append(ctypes.c_uint64(int(ptr)))
                    if name in self._outputs:
                        self._output_slots[name] = (ptr, host)
                elif name in self._shape.dims:
                    self._kernel_args.append(ctypes.c_int(int(self._shape.dims[name])))
                else:
                    raise BuildError(f"launch arg {name} is not an input, output, or shape dim")
        except BaseException:
            self._free()
            raise
        self.arg_ptrs = (ctypes.c_void_p * len(self._kernel_args))(
            *(ctypes.addressof(arg) for arg in self._kernel_args)
        )
        return self

    def copy_outputs_back(self) -> None:
        from cuda.bindings import driver as cuda

        for name, (ptr, host) in self._output_slots.items():
            _check(cuda.cuMemcpyDtoH(host.ctypes.data, ptr, host.nbytes))
            self._outputs[name][...] = host

    def __exit__(self, *_) -> None:
        self._free()

    def _free(self) -> None:
        from cuda.bindings import driver as cuda

        for ptr in self._allocations:
            cuda.cuMemFree(ptr)
        self._allocations = []


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

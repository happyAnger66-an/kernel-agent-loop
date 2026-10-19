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

    def load(self, artifact: Artifact):
        launch_config = _load_launch_config(artifact.source_dir / "launch.py")
        cubin = artifact.cubin_path.read_bytes()

        def run(inputs: dict, outputs: dict, shape) -> None:
            config = launch_config(shape)
            _launch(cubin, config, inputs, outputs, shape)

        return run


def _gpu_arch() -> str:
    from cuda.bindings import driver as cuda

    _check(cuda.cuInit(0))
    device = _check(cuda.cuDeviceGet(0))
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


def _launch(cubin: bytes, config: dict, inputs: dict, outputs: dict, shape) -> None:
    import ctypes

    from cuda.bindings import driver as cuda

    _check(cuda.cuInit(0))
    device = _check(cuda.cuDeviceGet(0))
    context = _check(cuda.cuCtxCreate(None, 0, device))
    allocations = []
    try:
        module = _check(cuda.cuModuleLoadData(cubin))
        function = _check(cuda.cuModuleGetFunction(module, config["kernel"].encode()))
        kernel_args = []
        output_slots = {}
        for name in config["args"]:
            if name in inputs or name in outputs:
                host = np.ascontiguousarray(inputs[name] if name in inputs else outputs[name])
                ptr = _check(cuda.cuMemAlloc(host.nbytes))
                allocations.append(ptr)
                _check(cuda.cuMemcpyHtoD(ptr, host.ctypes.data, host.nbytes))
                kernel_args.append(ctypes.c_uint64(int(ptr)))
                if name in outputs:
                    output_slots[name] = (ptr, host)
            elif name in shape.dims:
                kernel_args.append(ctypes.c_int(int(shape.dims[name])))
            else:
                raise BuildError(f"launch arg {name} is not an input, output, or shape dim")
        arg_ptrs = (ctypes.c_void_p * len(kernel_args))(*(ctypes.addressof(arg) for arg in kernel_args))
        grid, block = config["grid"], config["block"]
        _check(
            cuda.cuLaunchKernel(
                function,
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                int(config["smem"]),
                0,
                arg_ptrs,
                0,
            )
        )
        _check(cuda.cuCtxSynchronize())
        for name, (ptr, host) in output_slots.items():
            _check(cuda.cuMemcpyDtoH(host.ctypes.data, ptr, host.nbytes))
            outputs[name][...] = host
    finally:
        for ptr in allocations:
            cuda.cuMemFree(ptr)
        cuda.cuCtxDestroy(context)


def _check(result):
    from cuda.bindings import driver as cuda

    if isinstance(result, tuple):
        err, *rest = result
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise BuildError(str(err))
        return rest[0] if len(rest) == 1 else rest
    return result

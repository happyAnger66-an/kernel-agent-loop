"""模块: kernel_agent_loop.backends.cuda
功能: Thor 上把 CUDA 实现编成 cubin 并加载运行，编译错误带上 nvcc 输出。
"""

import shutil

import numpy as np
import pytest

pytestmark = pytest.mark.thor

if shutil.which("nvcc") is None:
    pytest.skip("nvcc not found", allow_module_level=True)
pytest.importorskip("cuda")

from kernel_agent_loop.backends.cuda import BuildError, CudaBackend  # noqa: E402
from kernel_agent_loop.core.opspec import Shape  # noqa: E402

VADD_CU = """
extern "C" __global__ void vadd(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}
"""

VADD_LAUNCH = """
def launch_config(shape):
    n = shape.dims["n"]
    return {"kernel": "vadd", "grid": ((n + 255) // 256, 1, 1), "block": (256, 1, 1),
            "smem": 0, "args": ["a", "b", "out", "n"]}
"""


@pytest.fixture
def impl_dir(tmp_path):
    (tmp_path / "kernel.cu").write_text(VADD_CU)
    (tmp_path / "launch.py").write_text(VADD_LAUNCH)
    return tmp_path


# 用例: 合法的 CUDA 实现能编译出 cubin 文件
# 参数: impl_dir 是含 kernel.cu 和 launch.py 的向量加法实现目录
def test_build_produces_a_cubin(impl_dir):
    artifact = CudaBackend().build(impl_dir)
    assert artifact.cubin_path.exists()
    assert artifact.cubin_path.suffix == ".cubin"


# 用例: 加载后的 kernel 用正确性检查的调用签名运行，结果等于两路输入相加
# 参数: impl_dir 是含 kernel.cu 和 launch.py 的向量加法实现目录
def test_loaded_kernel_runs_with_the_correctness_signature(impl_dir):
    backend = CudaBackend()
    run = backend.load(backend.build(impl_dir))
    shape = Shape("n1000", {"n": 1000})
    rng = np.random.default_rng(0)
    inputs = {"a": rng.standard_normal(1000, dtype=np.float32), "b": rng.standard_normal(1000, dtype=np.float32)}
    outputs = {"out": np.full(1000, np.nan, dtype=np.float32)}
    run(inputs, outputs, shape)
    np.testing.assert_allclose(outputs["out"], inputs["a"] + inputs["b"], rtol=1e-6)


# 用例: 源码编译失败时抛 BuildError，异常信息带上 nvcc 的报错
# 参数: impl_dir 是含 kernel.cu 和 launch.py 的向量加法实现目录
def test_compile_error_raises_build_error_with_nvcc_output(impl_dir):
    (impl_dir / "kernel.cu").write_text("this is not cuda\n")
    with pytest.raises(BuildError) as err:
        CudaBackend().build(impl_dir)
    assert "error" in str(err.value).lower()

"""模块: kernel_agent_loop.core.static_judge
功能: 拒绝实现目录以外的改动，以及进程、网络和文件逃逸。
"""

import pytest

from kernel_agent_loop.core.static_judge import StaticJudge

IMPL = "ops/rmsnorm/impls/cuda"
judge = StaticJudge(op="rmsnorm", backend="cuda")


# 用例: 只改实现目录里的 kernel 源码时检查通过
# 参数: 无
def test_clean_impl_change_passes():
    verdict = judge.check({f"{IMPL}/kernel.cu": "extern \"C\" __global__ void rmsnorm() {}\n"})
    assert verdict.ok, verdict.reasons


# 用例: 实现目录以外的改动一律拒绝，并在原因里点名该路径
# 参数: path 是越权文件的仓库相对路径，覆盖契约文件、测试文件和其他算子
@pytest.mark.parametrize("path", ["ops/rmsnorm/spec.py", "tests/evald/test_timing.py", "ops/softmax/impls/cuda/kernel.cu"])
def test_changes_outside_the_impl_dir_are_rejected(path):
    verdict = judge.check({f"{IMPL}/kernel.cu": "// ok\n", path: "x = 1\n"})
    assert not verdict.ok
    assert any(path in r for r in verdict.reasons)


# 用例: 实现目录里出现进程、网络或文件逃逸时拒绝
# 参数: path 是被检查的实现文件；content 是含 subprocess、socket、环境变量、读文件、fopen 或 system 的源码
@pytest.mark.parametrize("path,content", [
    (f"{IMPL}/launch.py", "import subprocess\n"),
    (f"{IMPL}/launch.py", "import socket\n"),
    (f"{IMPL}/launch.py", "eps = os.environ['EPS']\n"),
    (f"{IMPL}/launch.py", "data = open('/proc/self/maps').read()\n"),
    (f"{IMPL}/kernel.cu", "FILE *f = fopen(\"ref.bin\", \"rb\");\n"),
    (f"{IMPL}/kernel.cu", "system(\"ls\");\n"),
])
def test_io_and_process_escapes_are_rejected(path, content):
    verdict = judge.check({path: content})
    assert not verdict.ok

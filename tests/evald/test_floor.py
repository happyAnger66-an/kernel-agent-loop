"""模块: kernel_agent_loop.evald.floor
功能: 按 shape 字节数实测硬件下限（touch kernel），并写进 ShapePerf.floor_us。
"""

import shutil
from pathlib import Path

import pytest
from conftest import small_ops_root

from kernel_agent_loop.backends.cuda import Artifact
from kernel_agent_loop.core.opspec import load_opspec
from kernel_agent_loop.core.types import EnvSnapshot, Shape
from kernel_agent_loop.evald.evaluate import measure
from kernel_agent_loop.evald.floor import launch_config, shape_bytes

STEADY = EnvSnapshot(power_mode="MAXN", gpu_clock_mhz=1575, temp_c=50.0, throttled=False)


class ReferenceBackend:
    """不碰 GPU：build 原样返回，run 直接写参考答案。"""

    def __init__(self, spec) -> None:
        self.spec = spec

    def build(self, source_dir: Path) -> Artifact:
        return Artifact(cubin_path=Path(source_dir) / "kernel.cu", source_dir=Path(source_dir))

    def load(self, artifact: Artifact):
        def run(inputs, outputs, shape) -> None:
            for name, value in self.spec.reference(inputs).items():
                outputs[name][...] = value

        return run


# 用例: RMSNorm 的字节数 = x + weight 读入，y 写出
# 参数: repo_root 用来加载仓库里的 rmsnorm 契约
def test_shape_bytes_counts_inputs_and_outputs_once(repo_root):
    spec = load_opspec(repo_root / "ops", "rmsnorm")
    read, written = shape_bytes(spec, Shape("s", {"rows": 8, "hidden": 4096}))
    assert read == 8 * 4096 * 2 + 4096 * 2
    assert written == 8 * 4096 * 2


# 用例: 每线程一个 16 字节向量，grid 按较大一侧的元素数取整；0 字节时仍发射 1 个 block
# 参数: 无
def test_touch_launch_config_covers_larger_side():
    assert launch_config(Shape("f", {"n_in": 0, "n_out": 0}))["grid"][0] == 1
    assert launch_config(Shape("f", {"n_in": 512, "n_out": 256}))["grid"][0] == 2
    assert launch_config(Shape("f", {"n_in": 1 << 24, "n_out": 1 << 20}))["grid"][0] == 1 << 16


# 用例: 传入 floor_probe 时每个 shape 都带 floor_us，不传时为 None
# 参数: op_repo 是临时 git 仓库；repo_root 用来加载仓库里的 rmsnorm 契约
def test_measure_fills_floor_from_probe(op_repo, repo_root):
    spec = load_opspec(small_ops_root(repo_root), "rmsnorm")
    seen = []

    def probe(spec_, shape):
        seen.append(shape.label)
        return 3.0

    common = dict(repo=op_repo, op="rmsnorm", backend_name="cuda", commit="c", best="c", spec=spec,
                  backend=ReferenceBackend(spec), env_probe=lambda: STEADY, rounds=2, reps=1)
    result, _ = measure(**common, floor_probe=probe)
    assert seen == [s.label for s in spec.public_shapes()]
    assert all(perf.floor_us == 3.0 for perf in result.per_shape.values())
    result, _ = measure(**common)
    assert all(perf.floor_us is None for perf in result.per_shape.values())


@pytest.mark.thor
@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not found")
# 用例: 真 GPU 上下限随字节数单调增长，0 字节约等于空 kernel 发射开销
# 参数: 无
def test_floor_grows_with_bytes_on_gpu():
    pytest.importorskip("cuda")
    from kernel_agent_loop.evald.floor import FloorProbe

    probe = FloorProbe()
    try:
        empty = probe.floor_us(0, 0)
        small = probe.floor_us(1 << 16, 1 << 16)
        big = probe.floor_us(64 << 20, 64 << 20)
    finally:
        probe.close()
    assert 0 < empty < 20
    assert empty <= small * 1.2
    assert big > 10 * small

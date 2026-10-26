"""模块: kernel_agent_loop.evald.profile
功能: 在受限容器里用 ncu 采集候选 cubin 的指标；容器内不执行 Agent 的 Python。
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from conftest import small_ops_root

from kernel_agent_loop.backends.cuda import Artifact, BuildError, checked_launch_config
from kernel_agent_loop.core.opspec import load_opspec
from kernel_agent_loop.core.types import EnvSnapshot, Shape
from kernel_agent_loop.evald.evaluate import measure
from kernel_agent_loop.evald.profile import ProfileError, docker_command, parse_raw_csv, write_job

STEADY = EnvSnapshot(power_mode="MAXN", gpu_clock_mhz=1575, temp_c=50.0, throttled=False)
IMAGE = os.environ.get("KAL_PROFILE_IMAGE", "adas-img.nioint.com/r-lab/blackwell-aarch64:latest")

RAW = (
    '"ID","Process ID","Kernel Name","gpu__time_duration.sum","launch__registers_per_thread","mcc__x.pct"\n'
    '"","","","ns","register/thread","%"\n'
    '"0","11","rmsnorm","6,144","32","n/a"\n'
    '"1","11","rmsnorm","9000","40","12.5"\n'
    '"2","11","rmsnorm","nan","40","inf"\n'
)


WANTED = ("gpu__time_duration.sum", "mcc__x.pct", "missing__metric")


# 用例: raw CSV 按发射顺序对应 shape，跳过单位行，只留请求的指标，n/a 丢弃，千分位逗号去掉
# 参数: 无
def test_parse_raw_csv_maps_launches_to_labels():
    profile = parse_raw_csv("==PROF== noise\n" + RAW, ["a", "b", "c"], WANTED)
    assert profile["a"] == {"gpu__time_duration.sum": 6144.0}
    assert profile["b"] == {"gpu__time_duration.sum": 9000.0, "mcc__x.pct": 12.5}
    assert profile["c"] == {}


# 用例: 发射次数和 shape 数对不上时报错，不瞎对应
# 参数: 无
def test_parse_raw_csv_rejects_launch_count_mismatch():
    with pytest.raises(ProfileError):
        parse_raw_csv(RAW, ["a", "b"], WANTED)


# 用例: 容器只加 SYS_ADMIN、断网、只读根、作业目录只读挂载，不用 --privileged
# 参数: tmp_path 作业目录
def test_docker_command_is_locked_down(tmp_path):
    cmd = docker_command("docker", "img", tmp_path, ("m1", "m2"))
    joined = " ".join(cmd)
    assert "--privileged" not in cmd
    assert cmd[cmd.index("--cap-drop") + 1] == "ALL"
    assert cmd[cmd.index("--cap-add") + 1] == "SYS_ADMIN"
    assert cmd[cmd.index("--network") + 1] == "none"
    assert "--read-only" in cmd and "no-new-privileges" in cmd
    assert f"{tmp_path}:/job:ro" in cmd
    assert "--metrics m1,m2" in joined and "--clock-control none" in joined


# 用例: 作业目录里是 cubin、已求值的 launch 配置和输入数组，没有 launch.py
# 参数: tmp_path 作业目录；repo_root 用来加载 rmsnorm 契约
def test_write_job_contains_config_not_launch_py(tmp_path, repo_root):
    spec = load_opspec(repo_root / "ops", "rmsnorm")
    cubin = tmp_path / "k.cubin"
    cubin.write_bytes(b"\x7fELF")
    job = tmp_path / "job"
    job.mkdir()
    shape = Shape("r8", {"rows": 8, "hidden": 4096})
    config = {"kernel": "rmsnorm", "grid": [8, 1, 1], "block": [256, 1, 1], "smem": 0, "args": ["x"]}
    write_job(job, cubin, spec, [(shape, config)], seed=0)
    assert sorted(p.name for p in job.iterdir()) == ["kernel.cubin", "request.json", "shape0.npz"]
    request = json.loads((job / "request.json").read_text())
    assert request["shapes"][0]["config"] == config
    with np.load(job / "shape0.npz") as arrays:
        assert arrays["in:x"].shape == (8, 4096) and arrays["out:y"].shape == (8, 4096)


# 用例: launch 配置被规整成 JSON 类型；缺字段或 kernel 名不是标识符时报错
# 参数: 无
def test_checked_launch_config():
    ok = checked_launch_config({"kernel": "k", "grid": (2, 1, 1), "block": (64, 1, 1), "smem": 0, "args": ("x",)})
    assert ok["grid"] == [2, 1, 1] and ok["args"] == ["x"]
    with pytest.raises(BuildError):
        checked_launch_config({"kernel": "k; rm", "grid": (1, 1, 1), "block": (1, 1, 1), "smem": 0, "args": []})
    with pytest.raises(BuildError):
        checked_launch_config({"kernel": "k", "grid": (1, 1), "block": (1, 1, 1), "smem": 0, "args": []})


class ConfigBackend:
    """不碰 GPU：run 写参考答案，并像 CudaKernel 一样给出 launch_config。"""

    def __init__(self, spec) -> None:
        self.spec = spec

    def build(self, source_dir: Path) -> Artifact:
        return Artifact(cubin_path=Path(source_dir) / "kernel.cu", source_dir=Path(source_dir))

    def load(self, artifact: Artifact):
        spec = self.spec

        class Run:
            def __call__(self, inputs, outputs, shape):
                for name, value in spec.reference(inputs).items():
                    outputs[name][...] = value

            def launch_config(self, shape):
                return {"kernel": "k", "grid": [shape.dims["rows"], 1, 1]}

        return Run()


# 用例: 正确候选按 profile_shapes 送去 profile，结果进 EvalResult.profile；profiler 抛错时评测照常、原因进症状
# 参数: op_repo 是临时 git 仓库；repo_root 用来加载 rmsnorm 契约
def test_measure_profiles_requested_shapes_and_survives_failure(op_repo, repo_root):
    spec = load_opspec(small_ops_root(repo_root), "rmsnorm")
    target = spec.public_shapes()[:1]
    calls = []

    def profiler(cubin_path, spec_, items, *, seed):
        calls.append([(shape.label, config["grid"][0]) for shape, config in items])
        return {shape.label: {"gpu__time_duration.sum": 1.0} for shape, _ in items}

    def broken(*_, **__):
        raise ProfileError("no docker")

    common = dict(repo=op_repo, op="rmsnorm", backend_name="cuda", commit="c", best="c", spec=spec,
                  backend=ConfigBackend(spec), env_probe=lambda: STEADY, rounds=2, reps=1)
    result, _ = measure(**common, profiler=profiler, profile_shapes=target)
    assert calls == [[(target[0].label, target[0].dims["rows"])]]
    assert result.profile == {target[0].label: {"gpu__time_duration.sum": 1.0}}
    result, _ = measure(**common, profiler=broken, profile_shapes=target)
    assert result.correct and result.profile == {}
    assert any("profile unavailable: no docker" in s for s in result.symptoms)


def _image_available() -> bool:
    if shutil.which("docker") is None or shutil.which("nvcc") is None:
        return False
    return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


@pytest.mark.thor
@pytest.mark.skipif(not _image_available(), reason="docker, nvcc or the ncu image is missing")
# 用例: 真机上容器里的 ncu 对 RMSNorm 每个 shape 各采到一次发射的指标
# 参数: repo_root 提供 RMSNorm 实现与契约
def test_container_profile_rmsnorm_on_thor(repo_root):
    pytest.importorskip("cuda")
    from kernel_agent_loop.backends.cuda import CudaBackend
    from kernel_agent_loop.evald.profile import ContainerProfiler

    spec = load_opspec(repo_root / "ops", "rmsnorm")
    backend = CudaBackend()
    artifact = backend.build(repo_root / "ops/rmsnorm/impls/cuda")
    kernel = backend.load(artifact)
    shapes = spec.public_shapes()[:2]
    try:
        items = [(shape, kernel.launch_config(shape)) for shape in shapes]
    finally:
        kernel.close()
    profile = ContainerProfiler(image=IMAGE)(artifact.cubin_path, spec, items)
    assert list(profile) == [shape.label for shape in shapes]
    for shape in shapes:
        metrics = profile[shape.label]
        assert metrics["launch__grid_size"] == shape.dims["rows"]
        assert metrics["gpu__time_duration.sum"] > 0
        assert "smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio" in metrics

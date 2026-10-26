"""模块: kernel_agent_loop.evald.runner
功能: 子进程 Runner 通过 JSONL 执行 run，父进程持有输入输出缓冲。
"""

from pathlib import Path

import numpy as np

from kernel_agent_loop.backends.cuda import Artifact
from kernel_agent_loop.core.types import Shape
from kernel_agent_loop.evald.runner import SubprocessKernelRunner, materialize_artifact_dir


# 用例: fake worker 正确写回输出，且不改写输入
# 参数: tmp_path 是临时目录
def test_subprocess_runner_with_fake_worker_writes_outputs(tmp_path: Path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "kernel.cu").write_text("CANDIDATE_OK\n")
    (source / "launch.py").write_text("def launch_config(shape): return {}\n")
    artifact = Artifact(cubin_path=source / "kernel.cu", source_dir=source)
    dest = materialize_artifact_dir(artifact, tmp_path / "artifact")
    shape = Shape("r1-h8", {"rows": 1, "hidden": 8})
    inputs = {
        "x": np.ones((1, 8), dtype=np.float16),
        "weight": np.ones(8, dtype=np.float16),
    }
    outputs = {"y": np.full((1, 8), np.nan, dtype=np.float16)}
    original_x = inputs["x"].copy()
    with SubprocessKernelRunner(dest, worker_args=["--fake"]) as runner:
        runner.run(inputs, outputs, shape)
    assert np.array_equal(inputs["x"], original_x)
    assert np.isfinite(outputs["y"]).all()


# 用例: bench 在子进程内计时，只回传耗时；SLOW 标记的 0.02s 能被测出
# 参数: tmp_path 是临时目录
def test_subprocess_runner_bench_times_inside_child(tmp_path: Path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "kernel.cu").write_text("CANDIDATE_SLOW\n")
    (source / "launch.py").write_text("def launch_config(shape): return {}\n")
    artifact = Artifact(cubin_path=source / "kernel.cu", source_dir=source)
    dest = materialize_artifact_dir(artifact, tmp_path / "artifact")
    shape = Shape("r1-h8", {"rows": 1, "hidden": 8})
    inputs = {"x": np.ones((1, 8), dtype=np.float16), "weight": np.ones(8, dtype=np.float16)}
    outputs = {"y": np.full((1, 8), np.nan, dtype=np.float16)}
    with SubprocessKernelRunner(dest, worker_args=["--fake"]) as runner:
        us = runner.bench(inputs, outputs, shape)
    assert 20_000 <= us < 200_000
    assert np.isnan(outputs["y"]).all()


# 用例: fake worker 遇到 WRONG 标记时输出被清零
# 参数: tmp_path 是临时目录
def test_subprocess_runner_wrong_marker_zeros_output(tmp_path: Path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "kernel.cu").write_text("CANDIDATE_WRONG\n")
    (source / "launch.py").write_text("def launch_config(shape): return {}\n")
    artifact = Artifact(cubin_path=source / "kernel.cu", source_dir=source)
    dest = materialize_artifact_dir(artifact, tmp_path / "artifact")
    shape = Shape("r1-h4", {"rows": 1, "hidden": 4})
    inputs = {
        "x": np.ones((1, 4), dtype=np.float16),
        "weight": np.ones(4, dtype=np.float16),
    }
    outputs = {"y": np.full((1, 4), np.nan, dtype=np.float16)}
    with SubprocessKernelRunner(dest, worker_args=["--fake"]) as runner:
        runner.run(inputs, outputs, shape)
    assert np.all(outputs["y"] == 0)

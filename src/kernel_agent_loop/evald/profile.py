"""Nsight Compute profile of a candidate cubin, run in a locked-down container.

ncu needs CAP_SYS_ADMIN on Thor (RmProfilingAdminOnly=1), and a non-root user does not get it even with
--privileged. So ncu runs as root in a container with only SYS_ADMIN added, no network, a read-only root,
and the job dir mounted read-only. Only the trusted profile_target and the cubin run there: the
candidate's launch.py is evaluated beforehand in the isolated runner and passed in as JSON.

ncu replays each kernel, so gpu__time_duration here runs well above the bench timing (about 21 us vs
6 us for a one-block RMSNorm on Thor). Use the metrics to diagnose, never to compare speed.
"""

from __future__ import annotations

import csv
import io
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import kernel_agent_loop
from kernel_agent_loop.core.types import Shape
from kernel_agent_loop.evald.diagnose import STALLS

METRICS = (
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sectors_srcunit_tex_op_read.sum",
    "lts__t_sectors_srcunit_tex_op_write.sum",
    "lts__t_sectors_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__maximum_warps_per_active_cycle_pct",
    "sm__cycles_active.avg",
    "gpc__cycles_elapsed.max",
    "smsp__inst_executed.sum",
    "smsp__average_warp_latency_per_inst_issued.ratio",
    *(f"smsp__average_warp_latency_issue_stalled_{name}.ratio" for name in STALLS),
    "launch__registers_per_thread",
    "launch__grid_size",
    "launch__block_size",
    "launch__waves_per_multiprocessor",
    "launch__shared_mem_per_block_static",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__occupancy_limit_blocks",
    "device__attribute_multiprocessor_count",
)

CONTAINER_PKG = "/kal"
CONTAINER_JOB = "/job"


class ProfileError(Exception):
    pass


@dataclass(frozen=True)
class ContainerProfiler:
    image: str
    timeout: float = 600.0
    docker: str = "docker"
    metrics: tuple[str, ...] = METRICS

    def __call__(self, cubin_path: Path, spec, items: list[tuple[Shape, dict]], *, seed: int = 0
                 ) -> dict[str, dict[str, float]]:
        """{shape label: {metric: value}} for one launch of the cubin per (shape, launch config)."""
        job = Path(tempfile.mkdtemp(prefix="kal-profile-"))
        try:
            write_job(job, Path(cubin_path), spec, items, seed=seed)
            job.chmod(0o755)
            try:
                done = subprocess.run(
                    docker_command(self.docker, self.image, job, self.metrics),
                    capture_output=True, text=True, timeout=self.timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise ProfileError(f"ncu container timed out after {self.timeout:.0f}s") from exc
            if done.returncode != 0:
                raise ProfileError(f"ncu container exit {done.returncode}: {done.stderr.strip()[-2000:]}")
            return parse_raw_csv(done.stdout, [shape.label for shape, _ in items], self.metrics)
        finally:
            shutil.rmtree(job, ignore_errors=True)


def write_job(job: Path, cubin_path: Path, spec, items: list[tuple[Shape, dict]], *, seed: int) -> None:
    (job / "kernel.cubin").write_bytes(cubin_path.read_bytes())
    shapes = []
    for index, (shape, config) in enumerate(items):
        arrays = {f"in:{name}": value for name, value in spec.make_inputs(shape, seed=seed).items()}
        for name, (out_shape, dtype) in spec.output_specs(shape).items():
            arrays[f"out:{name}"] = np.zeros(out_shape, dtype=dtype)
        name = f"shape{index}.npz"
        np.savez(job / name, **arrays)
        shapes.append({"label": shape.label, "dims": dict(shape.dims), "config": config, "arrays": name})
    (job / "request.json").write_text(json.dumps({"shapes": shapes}))
    for path in job.iterdir():
        path.chmod(0o644)


def docker_command(docker: str, image: str, job: Path, metrics: tuple[str, ...]) -> list[str]:
    package = Path(kernel_agent_loop.__file__).resolve().parent
    ncu = (
        f"ncu --clock-control none --metrics {','.join(metrics)} -o /tmp/report "
        f"python3 -m kernel_agent_loop.evald.profile_target {CONTAINER_JOB} 1>&2 "
        "&& ncu --import /tmp/report.ncu-rep --csv --page raw --print-units base"
    )
    return [
        docker, "run", "--rm",
        "--runtime", "nvidia",
        "--cap-drop", "ALL",
        "--cap-add", "SYS_ADMIN",
        "--security-opt", "no-new-privileges",
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,size=512m",
        "--pids-limit", "256",
        "--memory", "8g",
        "-e", "HOME=/tmp",
        "-e", f"PYTHONPATH={CONTAINER_PKG}",
        "-v", f"{package}:{CONTAINER_PKG}/kernel_agent_loop:ro",
        "-v", f"{job}:{CONTAINER_JOB}:ro",
        "--entrypoint", "bash",
        image, "-c", ncu,
    ]


def parse_raw_csv(text: str, labels: list[str], metrics: tuple[str, ...]) -> dict[str, dict[str, float]]:
    """ncu --page raw rows in launch order, one per label, keeping only `metrics`; n/a values are dropped."""
    lines = [line for line in text.splitlines() if line.startswith('"')]
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    launches = [row for row in rows if row.get("ID", "").strip().isdigit()]
    if len(launches) != len(labels):
        raise ProfileError(f"ncu profiled {len(launches)} launches for {len(labels)} shapes")
    profile = {}
    for label, row in zip(labels, launches):
        values = {}
        for key in metrics:
            raw = row.get(key)
            if raw is None:
                continue
            try:
                value = float(str(raw).replace(",", ""))
            except ValueError:
                continue
            if math.isfinite(value):
                values[key] = value
        profile[label] = values
    return profile

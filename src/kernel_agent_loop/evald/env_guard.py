import shutil
import subprocess
from pathlib import Path

from kernel_agent_loop.core.types import EnvSnapshot

_NOT_THROTTLED = {"0", "0x0000000000000000", "Not Active", "[N/A]", "N/A", ""}
_GPU_GPC_FREQ = Path("/sys/class/devfreq/gpu-gpc-0/cur_freq")
_GPU_GPC_MAX = Path("/sys/class/devfreq/gpu-gpc-0/max_freq")


class EnvGuard:
    def __init__(self, max_temp_drift_c: float) -> None:
        self.max_temp_drift_c = max_temp_drift_c

    def check(self, before: EnvSnapshot, after: EnvSnapshot) -> list[str]:
        reasons = []
        if before.throttled or after.throttled:
            reasons.append("gpu throttled")
        if before.power_mode != after.power_mode:
            reasons.append(f"power mode changed from {before.power_mode} to {after.power_mode}")
        if before.gpu_clock_mhz != after.gpu_clock_mhz:
            reasons.append(f"gpu clock changed from {before.gpu_clock_mhz} to {after.gpu_clock_mhz}")
        drift = abs(after.temp_c - before.temp_c)
        if drift > self.max_temp_drift_c:
            reasons.append(f"temperature drifted {drift:.1f}°C")
        return reasons


def parse_gpu_env_line(line: str) -> EnvSnapshot:
    power, clock, temp, active = [part.strip() for part in line.split(",")]
    return EnvSnapshot(
        power_mode=power,
        gpu_clock_mhz=int(float(clock)),
        temp_c=float(temp),
        throttled=active not in _NOT_THROTTLED,
    )


def read_gpu_env() -> EnvSnapshot:
    """Desktop nvidia-smi first; Jetson Thor falls back when clocks report N/A."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi not found")
    completed = subprocess.run(
        [
            nvidia_smi,
            "--query-gpu=pstate,clocks.sm,temperature.gpu,clocks_event_reasons.active",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    line = completed.stdout.splitlines()[0]
    if _line_has_na(line) and _GPU_GPC_FREQ.is_file():
        return read_jetson_env(temp_c=_temp_from_smi_line(line))
    return parse_gpu_env_line(line)


def read_jetson_env(*, temp_c: float | None = None) -> EnvSnapshot:
    """Jetson/Thor probe: nvpmodel + gpu-gpc sysfs + nvidia-smi temperature."""
    if temp_c is None:
        temp_c = _read_gpu_temp_c()
    clock_hz = int(_GPU_GPC_FREQ.read_text().strip())
    max_hz = int(_GPU_GPC_MAX.read_text().strip()) if _GPU_GPC_MAX.is_file() else clock_hz
    return EnvSnapshot(
        power_mode=_nvpmodel_mode(),
        gpu_clock_mhz=clock_hz // 1_000_000,
        temp_c=float(temp_c),
        throttled=clock_hz < max_hz,
    )


def _line_has_na(line: str) -> bool:
    return any(part.strip() in {"[N/A]", "N/A"} for part in line.split(","))


def _temp_from_smi_line(line: str) -> float:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) >= 3 and parts[2] not in {"[N/A]", "N/A", ""}:
        return float(parts[2])
    return _read_gpu_temp_c()


def _read_gpu_temp_c() -> float:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError("nvidia-smi not found")
    completed = subprocess.run(
        [nvidia_smi, "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.splitlines()[0].strip())


def _nvpmodel_mode() -> str:
    nvpmodel = shutil.which("nvpmodel")
    if nvpmodel is None:
        return "unknown"
    completed = subprocess.run(
        ["sudo", "-n", nvpmodel, "-q"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        completed = subprocess.run([nvpmodel, "-q"], capture_output=True, text=True)
    for line in (completed.stdout or "").splitlines():
        if "NV Power Mode:" in line:
            return line.split(":", 1)[1].strip()
    return "unknown"

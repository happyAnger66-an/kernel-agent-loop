import shutil
import subprocess

from kernel_agent_loop.core.types import EnvSnapshot

_NOT_THROTTLED = {"0", "0x0000000000000000", "Not Active"}


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
    return parse_gpu_env_line(completed.stdout.splitlines()[0])

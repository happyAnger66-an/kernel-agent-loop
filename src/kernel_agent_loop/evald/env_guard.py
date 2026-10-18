from kernel_agent_loop.core.types import EnvSnapshot


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

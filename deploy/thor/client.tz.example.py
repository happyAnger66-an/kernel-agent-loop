"""Example: one optimization round against thor-003 via ssh nvidia@tz -p 2240."""

from pathlib import Path

from kernel_agent_loop.core.campaign import Campaign
from kernel_agent_loop.core.experiment_log import ExperimentLog
from kernel_agent_loop.core.gate import AcceptanceGate
from kernel_agent_loop.evald.client import EvalClient
from kernel_agent_loop.evald.env_guard import EnvGuard
from kernel_agent_loop.loop import run_round

REPO = Path(__file__).resolve().parents[2]
REMOTE_CONFIG = "/home/nvidia/kal/evald.config.json"
EVALD_CMD = "/home/nvidia/kal/venv/bin/python -m kernel_agent_loop.evald.serve"


def main() -> None:
    campaign = Campaign.load(REPO, "rms-thor")  # or Campaign.create(...)
    client = EvalClient(
        mode="ssh",
        host="tz",
        user="nvidia",
        port=2240,
        remote_config=REMOTE_CONFIG,
        evald_cmd=EVALD_CMD,
        remote_workdir="/home/nvidia/kal/inbox",
    )
    outcome = run_round(
        campaign,
        client=client,
        log=ExperimentLog(REPO / ".kal" / "rms-thor.sqlite"),
        gate=AcceptanceGate(env_guard=EnvGuard(max_temp_drift_c=5.0), min_gain_pct=1.0),
        lever="naive",
        hypothesis="thor smoke",
    )
    print(outcome.decision)


if __name__ == "__main__":
    main()

"""把工单和尝试归纳成 cell-result。不读日志，不跑评测。"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from jsonschema import Draft202012Validator

from kernel_agent_loop.core.experiment_log import Attempt
from kernel_agent_loop.knowledge.contracts import check_contracts, contracts_dir


class CellResultError(Exception):
    """格子结果对不上契约。"""


@dataclass(frozen=True)
class AttemptNote:
    """日志里没有、必须由调用方填写的尝试字段。e2e_ms 不从 per_shape 推。"""

    kind: str
    expected_gain_ms: float
    stage_reached: str
    e2e_ms: float | None = None
    note: str | None = None


def induce_cell_result(
    ticket: Mapping,
    attempts: Sequence[Attempt],
    notes: Sequence[AttemptNote],
    *,
    outcome: str,
    stop_reason: str,
    start: Mapping,
    final: Mapping | None = None,
    remaining: Sequence[Mapping] | None = None,
    conditions: Mapping | None = None,
    env: Mapping,
    finished_at: str,
) -> dict:
    """归纳一份格子结果。attempts 与 notes 按同一顺序对齐。"""
    check_contracts()
    if len(attempts) != len(notes):
        raise CellResultError("attempts 与 notes 数量必须相同")
    _check_outcome(outcome, stop_reason, attempts, final, remaining, conditions)
    result = {
        "schema_version": 1,
        "ticket": ticket.get("ticket"),
        "outcome": outcome,
        "stop_reason": stop_reason,
        "start": dict(start),
        "attempts": [_attempt_item(attempt, note) for attempt, note in zip(attempts, notes)],
        "env": dict(env),
        "finished_at": finished_at,
    }
    if final is not None:
        result["final"] = dict(final)
    if remaining is not None:
        result["remaining"] = [dict(item) for item in remaining]
    if conditions is not None:
        result["conditions"] = dict(conditions)
    validate_cell_result(result)
    return result


def _check_outcome(outcome, stop_reason, attempts, final, remaining, conditions) -> None:
    accepted = any(attempt.verdict == "accept" for attempt in attempts)
    if outcome == "apply":
        if final is None:
            raise CellResultError("apply 必须有 final")
        if not accepted:
            raise CellResultError("apply 的 attempts 至少一次 accept")
        return
    if outcome == "partial":
        if final is None:
            raise CellResultError("partial 必须有 final")
        if not remaining:
            raise CellResultError("partial 必须有非空 remaining")
        if not accepted:
            raise CellResultError("partial 的 attempts 至少一次 accept")
        return
    if outcome == "reject":
        software = conditions.get("software") if isinstance(conditions, Mapping) else None
        if not isinstance(software, Mapping) or not software:
            raise CellResultError("reject 必须有 conditions.software")
        if not attempts:
            raise CellResultError("reject 的 attempts 不能为空")
        if accepted:
            raise CellResultError("reject 的 attempts 不能有 accept")
        return
    if outcome == "measure":
        if attempts:
            raise CellResultError("measure 的 attempts 必须为空")
        if stop_reason != "measure_only":
            raise CellResultError("measure 的 stop_reason 必须是 measure_only")
        return
    raise CellResultError(f"outcome 非法: {outcome}")


def _attempt_item(attempt: Attempt, note: AttemptNote) -> dict:
    item = {
        "commit": attempt.commit,
        "parent": attempt.parent,
        "lever": attempt.lever,
        "kind": note.kind,
        "hypothesis": attempt.hypothesis,
        "expected_gain_ms": note.expected_gain_ms,
        "stage_reached": note.stage_reached,
        "verdict": attempt.verdict,
    }
    if note.e2e_ms is not None:
        item["e2e_ms"] = note.e2e_ms
    if note.note:
        item["note"] = note.note
    return item


def validate_cell_result(result: dict) -> None:
    schema = json.loads((contracts_dir() / "cell-result.schema.json").read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(result), key=lambda err: list(err.absolute_path))
    if not errors:
        return
    err = errors[0]
    path = ".".join(str(part) for part in err.absolute_path)
    where = f"{path}: " if path else ""
    raise CellResultError(f"格子结果未通过契约: {where}{err.message}")

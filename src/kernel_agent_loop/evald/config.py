"""Thor-side evald configuration (trusted tree, lock, hidden shapes)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kernel_agent_loop.core.types import Shape


@dataclass(frozen=True)
class EvaldConfig:
    trusted: Path
    work_root: Path
    lock_path: Path
    hidden_shapes: tuple[Shape, ...] = ()
    isolate: bool = True
    prepare_env: bool = False
    rounds: int = 5
    reps: int = 3
    run_timeout: float = 120.0
    ops_root: Path | None = None

    @classmethod
    def from_dict(cls, data: dict, *, base: Path | None = None) -> "EvaldConfig":
        base = Path(base or ".")
        hidden = tuple(_shape_from_dict(item) for item in data.get("hidden_shapes", []))
        hidden_path = data.get("hidden_shapes_path")
        if hidden_path:
            payload = json.loads(Path(hidden_path).expanduser().read_text())
            hidden = tuple(_shape_from_dict(item) for item in payload)
        ops_root = data.get("ops_root")
        return cls(
            trusted=_resolve(data["trusted"], base),
            work_root=_resolve(data["work_root"], base),
            lock_path=_resolve(data["lock_path"], base),
            hidden_shapes=hidden,
            isolate=bool(data.get("isolate", True)),
            prepare_env=bool(data.get("prepare_env", False)),
            rounds=int(data.get("rounds", 5)),
            reps=int(data.get("reps", 3)),
            run_timeout=float(data.get("run_timeout", 120.0)),
            ops_root=None if ops_root is None else _resolve(ops_root, base),
        )

    @classmethod
    def load(cls, path: Path) -> "EvaldConfig":
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text()), base=path.parent)


def _resolve(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _shape_from_dict(item: dict) -> Shape:
    return Shape(label=item["label"], dims=dict(item["dims"]))

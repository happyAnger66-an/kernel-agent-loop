import importlib.util
from pathlib import Path

from kernel_agent_loop.core.types import Shape

__all__ = ["Shape", "load_opspec"]


def load_opspec(ops_root: Path, name: str):
    path = Path(ops_root) / name / "spec.py"
    spec = importlib.util.spec_from_file_location(f"kernel_agent_loop_ops.{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load op spec {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SPEC

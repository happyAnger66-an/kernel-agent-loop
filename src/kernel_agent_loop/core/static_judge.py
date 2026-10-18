import re
from dataclasses import dataclass
from pathlib import PurePosixPath


_ESCAPE = re.compile(
    r"\bsubprocess\b|\bsocket\b|os\.environ|\bopen\s*\(|\bfopen\s*\(|\bsystem\s*\("
)


@dataclass(frozen=True)
class JudgeVerdict:
    ok: bool
    reasons: tuple[str, ...] = ()


class StaticJudge:
    def __init__(self, op: str, backend: str) -> None:
        self.impl_prefix = PurePosixPath("ops") / op / "impls" / backend

    def check(self, files: dict[str, str]) -> JudgeVerdict:
        reasons = []
        for path, content in files.items():
            posix = PurePosixPath(path)
            if ".." in posix.parts or not posix.is_relative_to(self.impl_prefix):
                reasons.append(f"path outside impl dir: {path}")
                continue
            if _ESCAPE.search(content):
                reasons.append(f"forbidden escape in {path}")
        if reasons:
            return JudgeVerdict(False, tuple(reasons))
        return JudgeVerdict(True)

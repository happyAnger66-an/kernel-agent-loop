import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


def git(cwd: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return out.stdout.strip()


@pytest.fixture
def op_repo(tmp_path: Path) -> Path:
    """A throwaway git repo shaped like this project, with one committed rmsnorm CUDA impl."""
    repo = tmp_path / "repo"
    impl = repo / "ops" / "rmsnorm" / "impls" / "cuda"
    impl.mkdir(parents=True)
    (repo / "ops" / "rmsnorm" / "spec.py").write_text("TRUSTED_SPEC\n")
    (impl / "kernel.cu").write_text("// v0\n")
    (impl / "launch.py").write_text("# v0\n")
    (repo / "README.md").write_text("readme\n")
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "test")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    return repo


_SMALL_OPS: list[Path] = []


def small_ops_root(repo_root: Path) -> Path:
    """Ops root whose rmsnorm spec keeps only the small public shapes, for fake-backend unit tests."""
    if not _SMALL_OPS:
        import tempfile

        root = Path(tempfile.mkdtemp(prefix="kal-small-ops-"))
        (root / "rmsnorm").mkdir()
        text = (repo_root / "ops" / "rmsnorm" / "spec.py").read_text()
        text += (
            "\n\nclass _SmallShapes(RmsNormSpec):\n"
            "    def public_shapes(self):\n"
            "        return [s for s in super().public_shapes() if s.dims['rows'] * s.dims['hidden'] <= 32768]\n"
            "\n\nSPEC = _SmallShapes()\n"
        )
        (root / "rmsnorm" / "spec.py").write_text(text)
        _SMALL_OPS.append(root)
    return _SMALL_OPS[0]

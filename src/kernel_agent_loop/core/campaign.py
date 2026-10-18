import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Campaign:
    repo: Path
    campaign_id: str
    op: str
    backend: str
    hardware: str
    best: str

    @classmethod
    def create(cls, repo: Path, *, campaign_id: str, op: str, backend: str, hardware: str) -> "Campaign":
        _git(repo, "checkout", "-b", f"kal/{campaign_id}")
        campaign = cls(Path(repo), campaign_id, op, backend, hardware, _git(repo, "rev-parse", "HEAD"))
        campaign._save()
        return campaign

    @classmethod
    def load(cls, repo: Path, campaign_id: str) -> "Campaign":
        data = json.loads(_meta_path(repo, campaign_id).read_text())
        return cls(Path(repo), campaign_id, data["op"], data["backend"], data["hardware"], data["best"])

    def commit_candidate(self, *, lever: str, hypothesis: str) -> str:
        impl = self._impl_dir()
        _git(self.repo, "add", "-A", "--", impl)
        _git(self.repo, "commit", "-m", f"{lever}\n\n{hypothesis}\n", "--", impl)
        return _git(self.repo, "rev-parse", "HEAD")

    def accept(self, sha: str) -> None:
        self.best = sha
        self._save()

    def revert(self) -> None:
        _git(self.repo, "checkout", self.best, "--", self._impl_dir())

    def _impl_dir(self) -> str:
        return f"ops/{self.op}/impls/{self.backend}"

    def _save(self) -> None:
        path = _meta_path(self.repo, self.campaign_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"op": self.op, "backend": self.backend, "hardware": self.hardware, "best": self.best}
            )
        )


def _meta_path(repo: Path, campaign_id: str) -> Path:
    return Path(repo) / ".git" / "kal" / f"{campaign_id}.json"


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return out.stdout.strip()

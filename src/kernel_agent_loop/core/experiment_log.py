import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Attempt:
    campaign: str
    commit: str
    parent: str
    lever: str
    hypothesis: str
    verdict: str
    symptoms: tuple[str, ...]
    per_shape: dict


class ExperimentLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._db = sqlite3.connect(self.path)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign TEXT NOT NULL,
                commit_sha TEXT NOT NULL,
                parent TEXT NOT NULL,
                lever TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                verdict TEXT NOT NULL,
                symptoms TEXT NOT NULL,
                per_shape TEXT NOT NULL
            )
            """
        )
        self._db.commit()

    def record(self, attempt: Attempt) -> None:
        self._db.execute(
            """
            INSERT INTO attempts
                (campaign, commit_sha, parent, lever, hypothesis, verdict, symptoms, per_shape)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt.campaign,
                attempt.commit,
                attempt.parent,
                attempt.lever,
                attempt.hypothesis,
                attempt.verdict,
                json.dumps(list(attempt.symptoms)),
                json.dumps(attempt.per_shape),
            ),
        )
        self._db.commit()

    def attempts(self, campaign: str) -> list[Attempt]:
        rows = self._db.execute(
            """
            SELECT campaign, commit_sha, parent, lever, hypothesis, verdict, symptoms, per_shape
            FROM attempts WHERE campaign = ? ORDER BY id
            """,
            (campaign,),
        ).fetchall()
        return [
            Attempt(
                campaign=row[0],
                commit=row[1],
                parent=row[2],
                lever=row[3],
                hypothesis=row[4],
                verdict=row[5],
                symptoms=tuple(json.loads(row[6])),
                per_shape=json.loads(row[7]),
            )
            for row in rows
        ]

    def tried(self, campaign: str, *, lever: str, parent: str) -> bool:
        row = self._db.execute(
            "SELECT 1 FROM attempts WHERE campaign = ? AND lever = ? AND parent = ? LIMIT 1",
            (campaign, lever, parent),
        ).fetchone()
        return row is not None

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArtifactEntry:
    path: str
    kind: str
    description: str


class ArtifactRegistry:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.entries: list[ArtifactEntry] = []

    def path(self, *parts: str) -> Path:
        target = self.root.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def add(self, path: Path | str, *, kind: str, description: str) -> None:
        target = Path(path)
        try:
            recorded = str(target.relative_to(self.root))
        except ValueError:
            recorded = str(target)
        self.entries.append(ArtifactEntry(path=recorded, kind=kind, description=description))

    def write_json(self, *parts: str, payload: Any, description: str) -> Path:
        target = self.path(*parts)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.add(target, kind="json", description=description)
        return target

    def write_text(self, *parts: str, text: str, description: str) -> Path:
        target = self.path(*parts)
        target.write_text(text, encoding="utf-8")
        self.add(target, kind="text", description=description)
        return target

    def write_manifest(self, *, metadata: dict[str, Any] | None = None) -> Path:
        payload = {
            "metadata": metadata or {},
            "artifacts": [asdict(entry) for entry in self.entries],
        }
        return self.write_json(
            "reports",
            "artifact_manifest.json",
            payload=payload,
            description="List of generated calculation artifacts.",
        )

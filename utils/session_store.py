from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from utils.path_safety import validate_request_id


class StoredSession(BaseModel):
    schema_version: int = 1
    request_id: str
    request: dict[str, Any]
    status: str = "UNKNOWN"
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    package: dict[str, Any] | None = None
    resume_state: dict[str, Any] = Field(default_factory=dict)
    resume_mode: Literal["none", "boundary", "replay"] = "none"


class SessionStore(Protocol):
    def save(self, session: StoredSession) -> None:
        ...

    def load(self, request_id: str) -> StoredSession | None:
        ...


class JsonSessionStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def save(self, session: StoredSession) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path_for(session.request_id)
        temp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temp_path.write_text(
                json.dumps(session.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def load(self, request_id: str) -> StoredSession | None:
        path = self._path_for(request_id)
        if not path.exists():
            return None
        try:
            return StoredSession.model_validate_json(path.read_text(encoding="utf-8"))
        except (JSONDecodeError, ValidationError, ValueError) as exc:
            raise CorruptSessionError(f"Stored session is corrupt: {path}") from exc

    def _path_for(self, request_id: str) -> Path:
        return self.directory / f"{validate_request_id(request_id)}.json"


class CorruptSessionError(RuntimeError):
    pass

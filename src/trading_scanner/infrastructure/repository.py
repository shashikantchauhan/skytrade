import asyncio
import json
from datetime import datetime
from pathlib import Path


class JsonSignalRepository:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    async def contains(self, fingerprint: str) -> bool:
        async with self._lock:
            return fingerprint in self._read()

    async def record(self, fingerprint: str, created_at: datetime) -> None:
        async with self._lock:
            signals = self._read()
            signals[fingerprint] = created_at.isoformat()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(signals, indent=2, sort_keys=True), encoding="utf-8")

    def _read(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text(encoding="utf-8"))

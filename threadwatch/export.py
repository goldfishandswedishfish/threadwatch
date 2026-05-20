from __future__ import annotations

import json
import dataclasses
from datetime import datetime
from pathlib import Path

from .trace import ProviderTrace


def _serialize(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def export_traces(task: str, traces: list[ProviderTrace]) -> Path:
    traces_dir = Path("traces")
    traces_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = traces_dir / f"{timestamp}.json"

    payload = {
        "task": task,
        "timestamp": timestamp,
        "providers": [dataclasses.asdict(t) for t in traces],
    }

    path.write_text(json.dumps(payload, indent=2, default=str))
    return path

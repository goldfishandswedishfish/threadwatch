from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

TOOLS_DIR = Path.home() / ".threadwatch" / "tools"


def _load_tools() -> tuple[list[dict], dict[str, object]]:
    definitions: list[dict] = []
    executors: dict[str, object] = {}
    if not TOOLS_DIR.exists():
        return definitions, executors

    for path in sorted(TOOLS_DIR.glob("*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"[threadwatch] warning: could not load {path.name}: {e}")
            continue
        definition = getattr(module, "DEFINITION", None)
        if not isinstance(definition, dict):
            continue
        name = definition.get("function", {}).get("name")
        executor = getattr(module, name, None) if name else None
        if name and executor:
            definitions.append(definition)
            executors[name] = executor

    return definitions, executors


TOOL_DEFINITIONS, _EXECUTORS = _load_tools()
TOOL_NAMES = {d["function"]["name"] for d in TOOL_DEFINITIONS}


async def execute_tool(name: str, arguments: dict) -> str:
    executor = _EXECUTORS.get(name)
    if executor is None:
        return f"Error: unknown tool '{name}'"
    try:
        result = executor(**arguments)
        if inspect.isawaitable(result):
            result = await result
        return str(result)
    except Exception as e:
        return f"Error running '{name}': {e}"

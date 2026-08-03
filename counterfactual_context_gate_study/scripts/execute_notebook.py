"""Execute plain-Python code cells from the checked-in CDCV notebook.

This dependency-free smoke runner is not a replacement for Jupyter execution;
it exists so CI can prove that every source cell runs in a clean shared
namespace without installing notebook packages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def execute(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__cdcv_notebook__"}
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        compiled = compile(source, f"{path.name}:cell-{index}", "exec")
        exec(compiled, namespace, namespace)
    return namespace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    namespace = execute(args.notebook.resolve())
    summary = namespace.get("smoke_summary")
    if not isinstance(summary, dict):
        raise RuntimeError("notebook did not produce smoke_summary")
    if summary.get("sealed_results_present") is not False:
        raise RuntimeError("notebook evidence lock is invalid")
    print("CDCV notebook smoke execution passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

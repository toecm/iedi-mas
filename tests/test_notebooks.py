from __future__ import annotations

import ast
import asyncio
import inspect
import json
from pathlib import Path


EXPECTED = {
    "Paper_2_IUUY.ipynb",
    "Paper_3_IEDI_MAS.ipynb",
    "Paper_4_JCCI.ipynb",
    "Paper_5_CA_IEDI.ipynb",
}


def test_paper_notebooks_are_clean_valid_json_and_compile() -> None:
    root = Path(__file__).parents[1]
    notebook_dir = root / "notebooks"
    assert {path.name for path in notebook_dir.glob("*.ipynb")} == EXPECTED

    for path in notebook_dir.glob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert notebook["cells"]
        source_text = ""
        for index, cell in enumerate(notebook["cells"]):
            source = "".join(cell["source"])
            source_text += source
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None
                assert cell["outputs"] == []
                compile(
                    source,
                    f"{path.name}:cell-{index}",
                    "exec",
                    flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                )
        assert "GOOGLE_API_KEY =" not in source_text
        assert "GEMINI_API_KEY =" not in source_text
        assert "google.generativeai" not in source_text
        assert "build_pipeline" in source_text


def test_paper_notebook_code_cells_execute_offline(monkeypatch) -> None:
    monkeypatch.delenv("IEDI_LIVE_GEMINI", raising=False)
    root = Path(__file__).parents[1]
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        namespace = {"__name__": f"offline_notebook_{path.stem}"}
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            compiled = compile(
                "".join(cell["source"]),
                f"{path.name}:cell-{index}",
                "exec",
                flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
            )
            outcome = eval(compiled, namespace)
            if inspect.isawaitable(outcome):
                asyncio.run(outcome)

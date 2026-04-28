"""Hygiene tests for the coursework notebook narrative.

These tests cover submission-facing notebook text rather than model outputs.
They focus on ensuring LungCancerML.ipynb no longer contains drafting prompts,
placeholder notes, or other unfinished coursework language.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "LungCancerML.ipynb"

PLACEHOLDER_PATTERNS = [
    r"\bTODO\b",
    r"buzz word soup",
    r"\blol\b",
    r"\bprobs\b",
    r"State what we want to learn",
    r"create sub-section",
    r"starting point\.",
    r"realist next steps",
    r"will need to justify",
    r"Any obvious issues from the first inspection\?",
    r"Any other features\?",
]


def load_markdown_text() -> str:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    markdown_chunks = [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "markdown"
    ]
    return "\n".join(markdown_chunks)


def test_notebook_markdown_has_no_drafting_placeholders() -> None:
    notebook_markdown = load_markdown_text()

    for pattern in PLACEHOLDER_PATTERNS:
        assert re.search(pattern, notebook_markdown, flags=re.IGNORECASE) is None

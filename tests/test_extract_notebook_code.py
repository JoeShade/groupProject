"""Unit tests for the notebook-code extraction helper.

These tests cover export sanitisation and notebook-structure validation.
They focus on the helper script in isolation so malformed notebook JSON fails
early without relying on blankTemplate.ipynb for reproduction.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_SCRIPT = REPO_ROOT / "scripts" / "extract_notebook_code.py"


def load_helper_module():
    spec = importlib.util.spec_from_file_location("extract_notebook_code", HELPER_SCRIPT)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def extractor_module():
    return load_helper_module()


def test_render_python_export_sanitises_magics_and_shell_lines(
    extractor_module,
    tmp_path: Path,
) -> None:
    notebook_path = tmp_path / "sample.ipynb"
    notebook_path.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "source": ["# Heading\n"]},
                    {
                        "cell_type": "code",
                        "source": [
                            "%matplotlib inline\n",
                            "x = 1\n",
                            "!echo hello\n",
                        ],
                    },
                    {
                        "cell_type": "code",
                        "source": [
                            "%%time\n",
                            "print('timed')\n",
                        ],
                    },
                    {
                        "cell_type": "code",
                        "source": (
                            "get_ipython().run_line_magic('matplotlib', 'inline')\n"
                            "y = 2\n"
                        ),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    exported_text = extractor_module.render_python_export(notebook_path)

    assert "Source notebook: sample.ipynb" in exported_text
    assert "# %% Notebook cell 2 (code cell 1)" in exported_text
    assert "# %matplotlib inline" in exported_text
    assert "x = 1" in exported_text
    assert "# !echo hello" in exported_text
    assert "# %%time" in exported_text
    assert "# print('timed')" in exported_text
    assert "# get_ipython().run_line_magic('matplotlib', 'inline')" in exported_text
    assert "y = 2" in exported_text


def test_load_notebook_rejects_missing_cells_list(
    extractor_module,
    tmp_path: Path,
) -> None:
    notebook_path = tmp_path / "invalid.ipynb"
    notebook_path.write_text(json.dumps({"metadata": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="cells"):
        extractor_module.load_notebook(notebook_path)


def test_normalise_source_lines_rejects_non_string_entries(extractor_module) -> None:
    with pytest.raises(TypeError, match="strings"):
        extractor_module.normalise_source_lines(["valid\n", 1])

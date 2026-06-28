"""Convert a percent-format ``.py`` into a Jupyter ``.ipynb`` (zero deps).

We author notebooks as plain Python in the "percent" cell format (``# %%`` for
code cells, ``# %% [markdown]`` for markdown), because raw ``.ipynb`` JSON is
heavy and painful to write/diff. This script is the build step that turns that
source into a runnable notebook. The generated ``.ipynb`` is a *build artifact*
and is git-ignored — commit the ``.py`` source instead.

Usage:
    python scripts/build_notebook.py notebooks/rotorenv_colab_train.py
    python scripts/build_notebook.py notebooks/foo.py --out /tmp/foo.ipynb

Cell format:
    # %%                 -> start a code cell
    # %% [markdown]      -> start a markdown cell (lines may be plain or '# '-prefixed)
"""

from __future__ import annotations

import argparse
import json
import os


def parse_cells(source: str) -> list[dict]:
    """Split percent-format source into notebook cell dicts.

    Args:
        source: The ``.py`` file contents.

    Returns:
        A list of nbformat cell dictionaries.
    """
    cells: list[dict] = []
    cell_type = "code"
    buf: list[str] = []

    def flush() -> None:
        # Drop leading/trailing blank lines, then emit if anything remains.
        text = "\n".join(buf).strip("\n")
        if not text.strip():
            return
        if cell_type == "markdown":
            # Strip a leading "# " from each line so prose reads as markdown.
            lines = [ln[2:] if ln.startswith("# ") else ln.lstrip("#").lstrip()
                     if ln.startswith("#") else ln for ln in text.split("\n")]
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": _as_source(lines)})
        else:
            cells.append({"cell_type": "code", "metadata": {},
                          "execution_count": None, "outputs": [],
                          "source": _as_source(text.split("\n"))})

    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush()
            buf = []
            cell_type = "markdown" if "[markdown]" in stripped else "code"
            continue
        buf.append(line)
    flush()
    return cells


def _as_source(lines: list[str]) -> list[str]:
    """Format a list of lines as nbformat source (newline-terminated except last)."""
    return [ln + "\n" for ln in lines[:-1]] + lines[-1:] if lines else []


def build(py_path: str, out_path: str | None) -> str:
    """Convert ``py_path`` to an ``.ipynb`` and write it.

    Args:
        py_path: Path to the percent-format source.
        out_path: Output path; defaults to the source with an ``.ipynb`` suffix.

    Returns:
        The path written.
    """
    with open(py_path) as f:
        source = f.read()
    notebook = {
        "cells": parse_cells(source),
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out_path = out_path or os.path.splitext(py_path)[0] + ".ipynb"
    with open(out_path, "w") as f:
        json.dump(notebook, f, indent=1)
    return out_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("py_path", help="percent-format .py source")
    parser.add_argument("--out", default=None, help="output .ipynb path")
    args = parser.parse_args()
    out = build(args.py_path, args.out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""Execute generated notebooks without requiring Jupyter or nbformat.

The runner supports the cell patterns used by this repository, captures text,
HTML tables, and Matplotlib figures, and writes standard notebook output JSON.
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import io
import json
import os
import pprint
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/llm-finance-matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def rich_output(value: Any) -> dict | None:
    if value is None:
        return None
    module = type(value).__module__
    if module.startswith("matplotlib"):
        return None
    if isinstance(value, pd.DataFrame):
        return {
            "output_type": "execute_result",
            "data": {
                "text/plain": value.to_string(max_rows=25, max_cols=20),
                "text/html": value.to_html(max_rows=25, max_cols=20, border=0),
            },
            "metadata": {},
        }
    if isinstance(value, pd.Series):
        return {
            "output_type": "execute_result",
            "data": {
                "text/plain": value.to_string(),
                "text/html": value.to_frame().to_html(border=0),
            },
            "metadata": {},
        }
    if hasattr(value, "to_html") and type(value).__name__ == "Styler":
        return {
            "output_type": "display_data",
            "data": {
                "text/plain": "<pandas.io.formats.style.Styler>",
                "text/html": value.to_html(),
            },
            "metadata": {},
        }
    if isinstance(value, (dict, list, tuple, set)):
        text = pprint.pformat(value, width=100, sort_dicts=True)
    else:
        text = repr(value)
    return {
        "output_type": "execute_result",
        "data": {"text/plain": text},
        "metadata": {},
    }


def figure_output(figure) -> dict:
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        metadata={"Software": "LLM in Finance notebook suite"},
    )
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "output_type": "display_data",
        "data": {
            "image/png": payload,
            "text/plain": f"<Figure size {figure.get_size_inches()[0]:.1f}x"
            f"{figure.get_size_inches()[1]:.1f} inches>",
        },
        "metadata": {},
    }


def execute_code(source: str, namespace: dict, count: int) -> list[dict]:
    outputs: list[dict] = []
    stdout = io.StringIO()
    stderr = io.StringIO()
    before = set(plt.get_fignums())

    def display(*values: Any) -> None:
        for value in values:
            output = rich_output(value)
            if output is not None:
                output["output_type"] = "display_data"
                output.pop("execution_count", None)
                outputs.append(output)

    namespace["display"] = display
    tree = ast.parse(source, mode="exec")
    last_expression = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expression = ast.Expression(tree.body.pop().value)
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if tree.body:
                exec(compile(tree, "<notebook>", "exec"), namespace)
            value = (
                eval(compile(last_expression, "<notebook>", "eval"), namespace)
                if last_expression is not None
                else None
            )
        if stdout.getvalue():
            outputs.insert(
                0,
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": stdout.getvalue(),
                },
            )
        if stderr.getvalue():
            outputs.append(
                {
                    "output_type": "stream",
                    "name": "stderr",
                    "text": stderr.getvalue(),
                }
            )
        output = rich_output(value)
        if output is not None:
            output["execution_count"] = count
            outputs.append(output)
        after = [number for number in plt.get_fignums() if number not in before]
        for number in after:
            outputs.append(figure_output(plt.figure(number)))
        for number in after:
            plt.close(number)
        return outputs
    except Exception as exc:
        tb = traceback.format_exc().splitlines()
        outputs.append(
            {
                "output_type": "error",
                "ename": type(exc).__name__,
                "evalue": str(exc),
                "traceback": tb,
            }
        )
        raise NotebookExecutionError(outputs) from exc


class NotebookExecutionError(RuntimeError):
    def __init__(self, outputs: list[dict]):
        super().__init__("notebook cell failed")
        self.outputs = outputs


def execute_notebook(source_path: Path, destination: Path) -> tuple[int, str | None]:
    notebook = json.loads(source_path.read_text(encoding="utf-8"))
    namespace = {
        "__name__": "__main__",
        "__file__": str(source_path),
        "ROOT": ROOT,
    }
    sys.path.insert(0, str(ROOT / "src"))
    old_cwd = Path.cwd()
    os.chdir(ROOT)
    count = 0
    error_message = None
    try:
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            count += 1
            source = "".join(cell["source"])
            try:
                cell["outputs"] = execute_code(source, namespace, count)
                cell["execution_count"] = count
            except NotebookExecutionError as exc:
                cell["outputs"] = exc.outputs
                cell["execution_count"] = count
                error = exc.__cause__
                error_message = (
                    f"cell {count}: {type(error).__name__}: {error}" if error else f"cell {count}"
                )
                break
    finally:
        os.chdir(old_cwd)
        if sys.path and sys.path[0] == str(ROOT / "src"):
            sys.path.pop(0)
        plt.close("all")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return count, error_message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--in-place", action="store_true")
    group.add_argument("--output-dir", type=Path)
    parser.add_argument("--pattern", default="*.ipynb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(NOTEBOOKS.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No notebooks matched {args.pattern!r}")
    failures = []
    for path in paths:
        destination = path if args.in_place else args.output_dir / path.name
        count, error = execute_notebook(path, destination)
        status = "PASS" if error is None else "FAIL"
        print(f"{status} {path.name} ({count} code cells)")
        if error:
            print(f"     {error}")
            failures.append((path.name, error))
    if failures:
        raise SystemExit(f"{len(failures)} notebook(s) failed")
    print(f"Executed {len(paths)} notebooks with zero cell errors.")


if __name__ == "__main__":
    main()


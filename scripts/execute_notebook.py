"""Execute a notebook sequentially in a fresh kernel and save its real outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

PROJECT = Path(__file__).resolve().parents[1]


def execute_notebook(path: Path, working_directory: Path) -> None:
    """Run cells in order and retain outputs beneath their cells."""
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    client = NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(working_directory)}},
    )
    client.create_kernel_manager()
    client.km.kernel_spec.argv[0] = sys.executable
    with client.setup_kernel():
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            try:
                client.execute_cell(cell, index)
                assert not any(output.output_type == "error" for output in cell.outputs)
                if cell.metadata.get("role") == "analysis":
                    assert len(cell.outputs) == 1, (index, len(cell.outputs))
                    data = cell.outputs[0].get("data", {})
                    assert "text/html" in data or "image/png" in data, index
                elif cell.metadata.get("role") == "technical":
                    assert not cell.outputs, index
            finally:
                # Preserve the last completed outputs, including errors if a run stops.
                nbformat.write(notebook, path)


def main() -> None:
    """Run one notebook from the packaged notebook directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook")
    args = parser.parse_args()
    for name in [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[name] = "1"
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    path = PROJECT / "notebooks" / args.notebook
    assert path.is_file() and path.suffix == ".ipynb"
    execute_notebook(path, PROJECT)
    print(f"Completed {path.name}")


if __name__ == "__main__":
    main()

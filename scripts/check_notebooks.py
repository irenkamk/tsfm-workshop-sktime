"""Execute solved copies of both notebooks without changing student notebooks."""
from pathlib import Path
import tempfile

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if "exercise" in cell.metadata.get("tags", []):
            cell.source = cell.metadata["solution"]
        if "reflection" in cell.metadata.get("tags", []):
            cell.source = cell.source.replace('= ""', '= "Validation: compare folds and note missing evidence."')
    with tempfile.TemporaryDirectory(prefix="tsfm-notebook-") as temporary:
        NotebookClient(
            notebook, timeout=180, kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        # Executed copies are deliberately temporary: no solutions/outputs
        # overwrite the student exercise cells in the distributed notebooks.
        nbformat.write(notebook, Path(temporary) / path.name)
    print(f"PASS: {path.name} (solutions, core execution, explicit missing-model path)")
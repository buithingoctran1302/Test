
from pathlib import Path
import subprocess
import sys
import nbformat
from nbclient import NotebookClient

# Thư mục gốc dự án
PROJECT_DIR = Path(__file__).resolve().parent.parent

SRC_DIR = PROJECT_DIR / "src"
NOTEBOOK_DIR = PROJECT_DIR / "notebooks"

EXECUTED_DIR = PROJECT_DIR / "reports" / "executed_notebooks"
EXECUTED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def run_notebook(relative_path):
    notebook_path = PROJECT_DIR / relative_path

    if not notebook_path.exists():
        raise FileNotFoundError(f"Không tìm thấy notebook: {notebook_path}")

    print(f"\n{'=' * 60}\nĐang chạy notebook: {notebook_path.name}")

    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_DIR)}},
    )
    client.execute()

    output_path = EXECUTED_DIR / f"{notebook_path.stem}_executed.ipynb"
    nbformat.write(notebook, output_path)

    print(f"Đã chạy xong: {notebook_path.name}")
    print(f"Notebook kết quả: {output_path}")


def run_python(relative_path):
    script_path = PROJECT_DIR / relative_path

    if not script_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file Python: {script_path}")

    print(f"\n{'=' * 60}\nĐang chạy Python: {script_path.name}")

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_DIR),
        check=True,
    )

    print(f"Đã chạy xong: {script_path.name}")


def main():
    steps = [
        ("Notebook", Path("src") / "data_preparation.ipynb"),
        ("Notebook", Path("notebooks") / "1.0-EDA.ipynb"),
        ("Python", Path("src") / "data_processing.py"),
        ("Python", Path("src") / "train_f3.py"),
        ("Python", Path("src") / "train_f5.py"),
    ]

    for file_type, relative_path in steps:
        if file_type == "Notebook":
            run_notebook(relative_path)
        else:
            run_python(relative_path)

    print("\nĐÃ CHẠY XONG TOÀN BỘ 5 BƯỚC!")


if __name__ == "__main__":
    main()
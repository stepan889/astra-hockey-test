"""Выполнить все ячейки ноутбука и сохранить видимые результаты."""

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


folder = Path(__file__).resolve().parent
os.chdir(folder)
path = folder / "hockey_solution.ipynb"
notebook = nbformat.read(path, as_version=4)

# Ядро использует тот же Python, которым запущен этот файл.
manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv[0] = sys.executable
client = NotebookClient(notebook, timeout=180, km=manager)
client.execute()
nbformat.write(notebook, path)
print("Готово:", path)

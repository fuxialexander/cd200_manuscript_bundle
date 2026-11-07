"""Packaged source and fixed model resources for the MoDL command-line workflow.

The original source tree and the installed CLI use the same unchanged engines.
Reference/ATAC stores and checkpoints remain separate scientific inputs.
"""
from pathlib import Path

def resource_path(relative: str = "") -> Path:
    root = Path(__file__).resolve().parent
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Resource path must stay inside the installed resource package")
    if not target.exists():
        raise FileNotFoundError(target)
    return target

def main():
    import runpy
    runpy.run_path(str(resource_path("scripts/modl.py")), run_name="__main__")

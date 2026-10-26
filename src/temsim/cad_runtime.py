"""Locations for the optional CAD process in source and wheel installations."""

import os
from pathlib import Path
import sys


def cad_process_root() -> Path:
    """Locate the runtime independently of editable configuration overrides."""
    source = Path(__file__).resolve()
    checkout = source.parents[2]
    if (source.relative_to(checkout).as_posix() == "src/temsim/cad_runtime.py"
            and (checkout / "pyproject.toml").is_file()):
        return checkout
    return Path(sys.prefix).resolve()


def _default_interpreter(root: Path) -> Path:
    executable = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return root / "environment" / "cad" / executable


def cad_interpreter(root: Path, override=None) -> Path:
    """Keep explicit interpreter selection ahead of the environment default."""
    configured = override if override is not None else os.environ.get("TEMSIM_CAD_PYTHON")
    if configured:
        return Path(configured).expanduser().resolve()
    return _default_interpreter(root)


def setup_instructions(root=None, interpreter=None):
    root = Path(root or cad_process_root()).resolve()
    executable = cad_interpreter(root, interpreter)
    invoke = "& " if os.name == "nt" else ""
    return (
        "Install the optional STEP geometry engine in a separate environment:\n"
        f'python -m venv "{root / "environment" / "cad"}"\n'
        f'{invoke}"{_default_interpreter(root)}" '
        f'-m pip install -r "{root / "requirements-cad.txt"}"\n'
        "Set TEMSIM_CAD_PYTHON to another CAD environment's Python executable if needed.\n"
        f"Selected CAD interpreter: {executable}\n"
        "The simulator's Python environment is unchanged."
    )

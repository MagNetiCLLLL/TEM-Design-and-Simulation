"""Source-checkout compatibility entry for the packaged CAD geometry worker."""

from pathlib import Path
import runpy


def main():
    entry = Path(__file__).resolve().parents[1] / "src" / "temsim" / "cad_geometry_worker.py"
    return runpy.run_path(str(entry))["main"]()


if __name__ == "__main__":
    raise SystemExit(main())

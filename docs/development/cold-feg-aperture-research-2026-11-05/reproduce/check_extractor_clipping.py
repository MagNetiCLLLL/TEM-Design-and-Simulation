"""Check extractor clipping with coordinate arrays, without physical transport."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from temsim.optics.electron_gun.electrostatic import ExtractorElectrode
from temsim.optics.electron_gun.tracing import _clip_body_bores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    profile = tomllib.loads(
        (REPO / "configs/instruments/gun/FEG.toml").read_text(encoding="utf-8")
    )
    part = next(row for row in profile["parts"] if row["key"] == "feg_extractor")
    extractor = ExtractorElectrode(
        mechanical_center_from_tip_mm=part["local_center_z_mm"],
        mechanical_length_mm=part["length_mm"],
        mechanical_outer_diameter_mm=part["outer_diameter_mm"],
        mechanical_clear_bore_diameter_mm=part["bore_diameter_mm"],
    )
    # Four synthetic coordinates; no source particles or trajectory integration.
    positions_mm = np.array([
        [1.9, 0.0, 8.0], [2.1, 0.0, 8.0],
        [2.1, 0.0, 5.0], [2.1, 0.0, 11.0],
    ])
    positions = positions_mm * 1e-3
    alive = np.ones(4, dtype=bool)
    blocked_z = np.full(4, np.nan)
    keys = [""] * 4
    _clip_body_bores(
        SimpleNamespace(bore_components=(extractor,)),
        positions.copy(), positions, alive, np.zeros(4, dtype=bool),
        blocked_z, keys,
    )
    assert alive.tolist() == [True, False, True, True]
    assert keys == ["", "feg_extractor", "", ""]
    assert blocked_z[1] == 8.0

    result = {
        "scope": "Pure clipping-helper coordinate arrays; no source emission, field solve or physical transport.",
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        ).strip(),
        "profile": "configs/instruments/gun/FEG.toml",
        "bore_diameter_mm": part["bore_diameter_mm"],
        "extent_mm": [part["local_start_z_mm"], part["local_end_z_mm"]],
        "cases": [
            {"r_mm": float(p[0]), "z_mm": float(p[2]),
             "passed": bool(passed), "blocked_key": key}
            for p, passed, key in zip(positions_mm, alive, keys)
        ],
        "passed": True,
        "limit": "Does not qualify continuous wall contact, transmitted current or energy filtering.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

"""Tune installed supplies on the common material mesh and relaunch the finite source."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "cescor_direct_probe_20261121"
sys.path.insert(0, str(PREVIOUS))
import material_relay_bore as bore
import material_relay_newton as newton
import material_relay_response as response
from material_relay_column import Column, Field, launch, source_and_electrodes
from material_relay_fields import WORK, digest
from material_relay_global import Global
from material_relay_newton import Circuit
from material_relay_run import static


PARENT = PREVIOUS / "evidence/material-relay/post-scan-objective-focus"
BASE = WORK / "global-23b8b2425e729ecb.npz"
OLD_GROUPS = response.groups
CONTROLS = {"c1": "condenser_lens_1/", "c2": "condenser_lens_2/",
            "adl": "adapter_lens", "tl21": "probe_tl21_lens", "tl22": "probe_tl22_lens"}


def groups(bias):
    return {**OLD_GROUPS(bias), **{key: [s for s in bias if s.startswith(prefix)]
                                   for key, prefix in CONTROLS.items()}}


def load_bias(path):
    with np.load(path) as data:
        result = json.loads(str(data["report"]))
    if result["residual"] > 2e-8:
        raise ValueError("The material solve has not converged")
    return result["excitation"]


def apply_supply_changes(bias, changes):
    drives = bias.copy()
    for name, current in changes.items():
        keys = groups(bias)[name]
        if not keys or not np.isfinite(current):
            raise ValueError(f"Invalid supply assignment {name}={current}")
        for key in keys:
            drives[key] = current
    return drives


def tangent(seed, names, spacing):
    with patch.object(response, "groups", groups):
        response.solve(Global(9), seed, names)
    with patch.object(bore, "groups", groups):
        bore.process(seed, names, spacing)


def run(seed, changes, label, count, step, spacing, predictor=None):
    drives = apply_supply_changes(load_bias(seed), changes)
    solution = WORK / ("global-" + digest(drives)[:16] + ".npz")
    if not solution.is_file():
        with patch.object(newton,"groups",groups):
            Circuit(Global(9)).solve(drives, seed, solution, predictor=predictor)
    if load_bias(solution) != drives:
        raise ValueError("Solved supply vector does not match the trial")
    bore.process(solution, ["dc"], spacing)
    field = Field(solution, step=spacing)
    source = source_and_electrodes()
    gun = launch(source, field, count)
    model = Column(source, field, gun, count, step,
                   observation_mm=(764.9, 765.1, 1179.1, 1237.2, 1420., 1512.5))
    folder = HERE / label
    folder.mkdir(exist_ok=True)
    report = static(model, gun, folder, True, "Coupled winding and magnetic-material FEM")
    with np.load(folder / "column-and-landings.npz") as data:
        loss = data["loss"]
        if np.any(loss < -1):
            raise ValueError("A particle left the verified numerical field")
        report["loss_count"] = int(np.count_nonzero(loss >= 0))
    report.update(field=field.receipt, currents=changes, gun_source=str(gun), step_mm=step,
                  solution=str(solution), bore_step_mm=spacing, baseline=str(PARENT))
    (folder / "tracking.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Completed", label, solution.name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("tangent", "run"))
    parser.add_argument("--seed", type=Path, default=BASE)
    parser.add_argument("--predictor", type=Path)
    parser.add_argument("--keys", nargs="+")
    parser.add_argument("--set", nargs="+", default=[])
    parser.add_argument("--label", default="objective-only")
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--step", type=float, default=.075)
    parser.add_argument("--spacing", type=float, default=.25)
    args = parser.parse_args()
    if args.task == "tangent":
        tangent(args.seed, args.keys, args.spacing)
    else:
        run(args.seed, {s.split("=")[0]: float(s.split("=")[1]) for s in args.set},
            args.label, args.count, args.step, args.spacing, args.predictor)

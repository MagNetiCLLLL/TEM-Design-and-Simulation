"""Retune the output relay with the fixed +50 V FEI gun and common nonlinear circuit."""

from pathlib import Path
import json
import sys

import numpy as np
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
COND = HERE.parent / "fei-condenser"
sys.path.insert(0, str(COND))
from column_trace import BoreTracker, covariance, optics_mesh, propagate, source_packet, values, wq
from relaunch import relaunch
from volume import CACHE, Vacuum, layered, sample
from material_trial import Circuit, Global, apply_supply_changes, digest, groups, load_bias, newton
from material_relay_fields import WORK, sha
from relay_transfer import advance, electric_reference

BASE = WORK / "global-b7e03e3ec2da7af4.npz"
TL = load_bias(BASE)["probe_tl12_lens"]
OBJ = load_bias(BASE)["objective_lens/physical-pack-4"]
MIX = np.array(
    json.loads((HERE.parent / "c1-zoom-focused/ac-pivot.json").read_text())["pivot"][
        "physical_A_per_command_urad"
    ]
)
CHANNELS = [f"ac_{station}_{axis}" for station in (0, 1) for axis in "xy"]


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checkpoint():
    target = CACHE / "fei-output-b7e03e3ec2da7af4-handoff-1320.npz"
    if not target.exists():
        packet = relaunch(BASE, 524288)
        m, _, _ = optics_mesh(0.06)
        t = BoreTracker(m, layered(BASE), source_packet(packet))
        i = np.searchsorted(m.z, 1.320)
        ((_, p, loss),) = propagate(t, m, [i])
        keep = loss == -1
        np.savez_compressed(
            target,
            z=m.z[i],
            p=p[keep],
            energy=300000 + t.de[keep],
            potential=electric_reference(t)[keep],
            w=t.weights[keep] * 15,
            ids=t.ids[keep],
            field_sha256=sha(BASE),
            original_count=524288,
        )
    return target


def axis(path, z):
    out = CACHE / (path.stem + "-output-axis.npz")
    if not out.exists():
        grid = np.linspace(1.295, 1.613, 1591)
        b = Vacuum(path).direct(np.column_stack([grid * 0, grid * 0, grid]))[:, 2]
        np.savez(out, z=grid, b=b, sha256=sha(path))
    with np.load(out) as f:
        return CubicSpline(f["z"], f["b"])(z)


if __name__ == "__main__":
    print("DC controls", TL, OBJ, flush=True)
    print("Entrance cohort", checkpoint(), flush=True)

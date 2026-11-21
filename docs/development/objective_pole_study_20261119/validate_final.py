"""Compare matched high-angle GPU phases and individual contacts to the CPU RK implementation."""

import json
import argparse

import numpy as np
from numba import set_num_threads

from column import CACHE, Mesh, SURFACE, metrics
from particles import Tracker
from stationary_pupil import _transport, kinematics


def check(case, tag):
    fit = json.loads((CACHE / f"fit-{case}-{tag}.json").read_text())
    m = Mesh(0.1, [SURFACE])
    m.hex *= fit["hex_scale"]
    t = Tracker(m)
    f = m.field(m.base(case, fit["ni"], fit["level"], fit["padding"]), fit["controls"])
    result, loss, _, _ = t.run(f)
    blocked = np.flatnonzero(loss >= 0)
    clear = np.flatnonzero(loss < 0)
    # This numerical equivalence sample deliberately includes every physical loss.
    selected = np.r_[blocked, clear[np.linspace(0, len(clear) - 1, 1024 - len(blocked)).astype(int)]]
    cpu_losses = []
    errors = []
    batch_size = 128
    for k in range(0, len(selected), batch_size):
        ix = selected[k : k + batch_size]
        en = 300000.0 + t.de[ix]
        cpu, _ = _transport(
            t.input[ix], f, np.diff(m.z), *kinematics(en), np.arange(len(m.z)), False, m.electric, en
        )
        radii = np.hypot(cpu[:, :, 0], cpu[:, :, 1])
        hit = np.maximum(radii[:-1], radii[1:]) > m.radius[:, None]
        cpu_losses.extend(np.where(hit.any(axis=0), np.argmax(hit, axis=0), -1))
        surviving = loss[ix] < 0
        if np.any(surviving):
            errors.append(np.max(abs(cpu[-1, surviving] - result[-1, ix[surviving]]), axis=0))
    cpu_losses = np.asarray(cpu_losses)
    assert np.array_equal(cpu_losses, loss[selected])
    error = np.max(errors, axis=0)
    assert max(error[:2]) < 1e-12 and max(error[2:4]) < 1e-9
    original_corrector = Mesh(0.1, [SURFACE])
    restored = original_corrector.field(
        original_corrector.base(case, fit["ni"], fit["level"], fit["padding"]), fit["controls"]
    )
    alternate, stopped, _, _ = t.run(restored)
    active = stopped == -1
    return dict(
        count=len(selected),
        covered_all_losses=len(blocked),
        same_first_contact_step=True,
        absolute_phase_errors=error.tolist(),
        unchanged_controls_saved_hexapoles=metrics(alternate[-1, active], t.weights[active], t.current),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="jointfine")
    args = p.parse_args()
    set_num_threads(12)
    result = {case: check(case, args.tag) for case in ["photo", "photo_iron_tube"]}
    print(result, flush=True)
    (CACHE / "matched-transport-check.json").write_text(json.dumps(result, indent=2))

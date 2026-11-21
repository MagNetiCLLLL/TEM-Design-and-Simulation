"""Independent CPU Lorentz reference and parked control on the original particles."""

import json

import numpy as np
from numba import set_num_threads

from column import CACHE, Mesh, SURFACE, metrics
from particles import Tracker
from stationary_pupil import _transport, kinematics


if __name__ == "__main__":
    set_num_threads(12)
    results = {}
    m = Mesh(0.125, [SURFACE])
    t = Tracker(m)
    for case in ("parked", "photo"):
        field = m.field(m.base(case, 3000.0, "fine", 8), [1.0, 1.0, 1.0])
        observed, loss, peak, margin = t.run(field)
        assert np.all(loss == -1)
        en = 300000.0 + t.de[:1024]
        cpu, _ = _transport(
            t.input[:1024], field, np.diff(m.z), *kinematics(en), m.saved, False, m.electric, en
        )
        errors = np.max(abs(observed[-1, :1024] - cpu[-1]), axis=0)
        result = metrics(observed[-1], t.weights, t.current)
        results[case] = dict(
            probe=result,
            CPU_subset=1024,
            absolute_phase_error=errors.tolist(),
            largest_radius_mm=float(peak.max() * 1000),
            min_clearance_mm=float(margin.min() * 1000),
        )
        print(case, results[case], flush=True)
    (CACHE / "transport-check.json").write_text(json.dumps(results, indent=2))

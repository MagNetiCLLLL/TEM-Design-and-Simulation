"""Identify which inherited coils produce multiple waists near the gun exit."""

import argparse

import numpy as np

from column_current import ACTIVE, OWNER, source_digest
from joint_pivot_tuning import stations
from joint_relay_search import NI, model_at, mono_roots
from normal_working_point import EVIDENCE
from relay_transfer import advance
from trace_normal_joint_field import beam_moments
from verify_joint_alignment import keep_json


def check(amps):
    model, q = model_at(amps, NI, 32768, 0.0375, 24)
    c1 = OWNER[ACTIVE] == "condenser_lens_1"
    c2 = OWNER[ACTIVE] == "condenser_lens_2"
    stations_m = stations(model, np.arange(0.450, 0.5501, 0.000125))
    physical = model.m.nodes
    interval = (physical >= stations_m[0]) & (physical <= stations_m[-1])
    rows = dict(
        z_interval_mm=(stations_m[[0, -1]] * 1e3).tolist(),
        source_digest=source_digest(model.t),
        source_count=len(model.loss),
        step_mm=model.step,
    )
    for lens, mask in (("C1", c1), ("C2", c2)):
        component = np.einsum("i,ij->j", q[mask], model.coils[mask, :, 0])
        rows[f"peak_{lens}_Bz_T"] = float(abs(component[interval]).max())
    for name, switched_off in (
        ("installed", np.zeros_like(c1)),
        ("C1_off", c1),
        ("C2_off", c2),
        ("C1_and_C2_off", c1 | c2),
    ):
        field = model.field(np.where(switched_off, 0.0, q))
        p, loss, first = model.t.input, model.loss, 0
        moments = []
        indices = model.indices(stations_m)
        for group in np.array_split(indices, (len(indices) + 15) // 16):
            for p, loss in advance(model.t, field, p, model.en, model.phi0, loss, first, group):
                if np.any(loss != -1):
                    raise ValueError("Physical interception changed the upstream comparison cohort")
                moments.append(beam_moments(p, loss == -1, model.t.weights))
            first = group[-1]
        values = np.asarray(moments)
        roots = mono_roots(stations_m * 1000, values[:, 1] + values[:, 2])
        rows[name] = dict(
            radial_mm=roots.tolist(),
            principal_counts=[len(mono_roots(stations_m * 1000, values[:, i])) for i in (3, 4)],
            live=len(loss),
            min_max_rms_um=[float(values[:, 0].min()), float(values[:, 0].max())],
        )
        print(name, len(roots), flush=True)
    keep_json(EVIDENCE / "joint-c1-topology-control.json", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amps", type=float, default=255.71838)
    check(parser.parse_args().amps)

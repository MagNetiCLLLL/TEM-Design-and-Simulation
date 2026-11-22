"""Reduce measured annular landing errors with the installed round-lens subwindings."""

import argparse
import json
from pathlib import Path

import numpy as np

from relay_transfer import FOIL, HP1, HP2, RADIUS, SURFACE, Native, save
from temsim.physics.beam_statistics import transverse_beam_statistics


def run(a):
    seed = json.loads(a.settings.read_text())
    q = np.array(seed["currents_A"])
    planes = [HP2, HP1, a.pupil * .001]
    model = Native(a.table, .10, a.count, seed["objective_At"], planes)
    scale = np.array([350] * 4 + [220] * 4 + [115] * 4 + [450] * 3 + [600] * 3 + [400] * 3)
    _, (incident, _) = model.replay(q, [FOIL[0], SURFACE])
    radii = np.linalg.norm(incident[0][:, :2], axis=1) / RADIUS
    bins = np.digitize(radii, [0.20, 0.33, 0.45, 0.58, 0.72])
    target_foil = None

    def measure(x):
        nonlocal target_foil
        f, (front, (sample, loss)) = model.replay(x, [FOIL[0], SURFACE])
        incoming, contact = front
        live = loss == -1
        w = model.t.weights
        foil = transverse_beam_statistics(*incoming[contact == -1, :4].T, weights=w[contact == -1])
        foil_d95 = foil.radius_95_m * 2e6
        if target_foil is None:
            target_foil = foil_d95
        relay = model.central_relay(f, planes)
        edge = model.edge(f, model.entrance(f, front), azimuths=24)
        if edge["reached_sample"] != 24 or live.sum() < .995 * len(live):
            raise ValueError("Vignetting of the fixed pupil or source during the trial")
        pupil = incoming[:, 0] + 1j * incoming[:, 1]
        position = (sample[:, 0] + 1j * sample[:, 1]) * 1e6
        center = np.average(position[live], weights=w[live])
        landing = (position - center) * np.exp(-1j * np.angle(pupil))
        annular = np.array([np.average(landing[live & (bins == i)], weights=w[live & (bins == i)])
                            for i in range(6)])
        station = np.array([v["signed_B_mm"] for v in relay] +
                           [edge["angle_mrad_min_median_max"][-1] - 30, foil_d95 - target_foil])
        spread = np.percentile(abs(position[live] - center), 95) * 2
        return np.r_[station / [.045, .045, .055, .085], annular.real, annular.imag], spread

    residual, diameter = measure(q)
    damping = .1
    for i in range(a.steps):
        score = np.linalg.norm(residual)
        save(a.name + f"-{i:02}", dict(currents_A=q.tolist(), objective_At=model.ni, D95_um=diameter,
                                     residual=residual.tolist(), incoming_D95_um=target_foil))
        print(i, "D95 (um), score", diameter, score, flush=True)
        perturb = np.array([scale[j] * np.eye(len(q))[j] * .00004 for j in range(len(q))])
        jac = np.column_stack([(measure(q + dp)[0] - residual) / .00004 for dp in perturb])
        left, singular, right = np.linalg.svd(jac, full_matrices=False)
        for _ in range(6):
            dx = -right.T @ ((left.T @ residual) * singular / (singular**2 + damping**2))
            dx *= min(1, .25 / max(1e-12, abs(dx).max()))
            for fraction in (1, .45, .20, .075):
                trial = q + fraction * scale * dx
                try:
                    fresh, trial_d = measure(trial)
                except ValueError:
                    continue
                if np.linalg.norm(fresh) < score - .0001:
                    residual, q, diameter = fresh, trial, trial_d
                    damping = max(.025, damping * .55)
                    break
            else:
                damping *= 3
                continue
            break
        else:
            return
    save(a.name + "-final", dict(currents_A=q.tolist(), objective_At=model.ni, D95_um=diameter,
                                 residual=residual.tolist(), incoming_D95_um=target_foil))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name")
    p.add_argument("--settings", type=Path, required=True)
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--pupil", type=float, default=1408.05425036)
    p.add_argument("--count", type=int, default=4096)
    p.add_argument("--steps", type=int, default=10)
    run(p.parse_args())

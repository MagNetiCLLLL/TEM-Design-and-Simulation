"""Measure the illuminated aperture and test the shallow relay minimum against source sampling."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from column_current import C2_FOIL
from condenser_source_fields import write
from joint_relay_search import mono_roots
from material_relay_column import Column, Field, source_and_electrodes
from material_relay_fields import SAVE, WORK
from relay_transfer import advance
from trace_normal_joint_field import beam_moments


def measure(folder):
    r = json.loads((folder / "tracking.json").read_text())
    source = Path(r["gun_source"])
    field = Field(WORK / r["field"]["source"], step=r["field"]["step_mm"])
    column = Column(source_and_electrodes(), field, source, r["count"], r["step_mm"])
    tracker = column.t
    front = r["isolated_pair_fronts"][0]["z_mm"]
    aperture_stops = column.indices(C2_FOIL)
    region_stops = column.indices(np.arange(1420., 1512.51, .5) * .001)
    # These disjoint batches contain only original rays. Their spread measures dependence
    # on finite sampling of the same source; it is not a statistical confidence interval.
    batches = np.array_split(np.arange(r["count"]), 4)
    if min(map(len, batches)) < 1000:
        raise ValueError("Each comparison requires at least 1000 original source particles")
    populations = [np.arange(r["count"]), *batches]
    growth = [[] for batch in populations]
    aperture = []
    phase, loss, previous = tracker.input, column.loss, 0
    for group in np.array_split(np.r_[aperture_stops, region_stops], 16):
        for stop, (phase, loss) in zip(
            group, advance(tracker, column.full, phase, column.en, column.phi0, loss, previous, group),
            strict=True
        ):
            live = loss == -1
            if stop in aperture_stops:
                radius = np.linalg.norm(phase[live, :2], axis=1) * 1e6
                w = tracker.weights[live]
                order = np.argsort(radius)
                i = np.searchsorted(np.cumsum(w[order]), .95 * w.sum())
                aperture.append(dict(z_mm=1000 * column.m.z[stop], radius_um=50.,
                                     populated_radius95_um=radius[order[i]], maximum_radius_um=radius.max(),
                                     surviving=int(live.sum()), transmitted_A=float(tracker.current * w.sum())))
                continue
            for population, values in zip(populations, growth, strict=True):
                m = beam_moments(phase[population], live[population], tracker.weights[population])
                values.append([m[0], m[1]+m[2], m[3], m[4]])
        previous = group[-1]
    z = column.m.z[region_stops] * 1000
    results = []
    for population, values in zip(populations, growth, strict=True):
        values = np.array(values)
        at_front = CubicSpline(z, values)(front)
        results.append(dict(original_particles=len(population), first_id=int(tracker.ids[population[0]]),
                            last_id=int(tracker.ids[population[-1]]), front_rms_nm=1000 * at_front[0],
                            front_half_variance_slopes_nm2_per_mm=(1e6 * at_front[1:]).tolist(),
                            radial_minima_mm=mono_roots(z, values[:,1]).tolist(),
                            directional_minima_mm=[mono_roots(z, values[:,j]).tolist() for j in (2,3)]))
    write(folder / "plane-check.json", dict(field_sha256=r["field"]["solution_sha256"], count=r["count"],
          focal_plane_mm=front, C2_aperture_faces=aperture, examined_mm=[z[0],z[-1]],
          full_population=results[0], disjoint_original_source_quarters=results[1:]))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    measure(SAVE / parser.parse_args().label)

"""Locate D95 and RMS waists of the same transmitted primaries in the objective field."""

from pathlib import Path
import json
import numpy as np
from scipy.interpolate import CubicSpline
from temsim.physics.fei_tip_transport import REST_EV
from temsim.physics.joint_fem_column import trace_column
from temsim.joint_fem import input_file


def widths_um(xy, weights):
    xy = np.asarray(xy, dtype=float)
    delta = xy - np.average(xy, axis=0, weights=weights)
    r = np.linalg.norm(delta, axis=1)
    order = np.argsort(r)
    k = np.searchsorted(np.cumsum(weights[order]), 0.95 * weights.sum())
    rms_xy = np.sqrt(np.average(delta * delta, axis=0, weights=weights))
    return [2 * r[order[k]], np.hypot(*rms_xy), *rms_xy]


def measure_waists(folder, local_electric):
    folder = Path(folder)
    profile = json.loads((folder / "profile.json").read_text())
    with (
        np.load(input_file(profile, "tracking")) as m,
        np.load(input_file(profile, "column_magnetic")) as table,
        np.load(folder / "trajectories.npz") as d,
        np.load(local_electric) as local,
    ):
        position = float(d["z_mm"][-1])
        initial = np.c_[d["sample_xy_um"] * 1e-6, d["sample_slopes"]]
        weights = d["sample_weights_uA"]
        ids = d["sample_ids"]
        gun = d["gun_exit_phase"][d["column_loss"] == -1]
        np.testing.assert_array_equal(d["gun_exit_ids"][d["column_loss"] == -1], ids)
        np.testing.assert_array_equal(d["emitter_weights_uA"][ids], weights)
        factor = (
            -2 * float(m["reference_factor"]) * np.sqrt(300000 * (300000 + 2 * REST_EV))
        )
        u2 = (gun[:, 3:6] ** 2).sum(axis=1)
        energy = REST_EV * u2 / (np.sqrt(1 + u2) + 1)
        scales = table["scales"]
        magnetic = CubicSpline(
            table["z_m"], table["layers"].swapaxes(0, 1), axis=0, extrapolate=False
        )
        zv = local["z"]
        r = m["electric_r"]
        r = r[: np.searchsorted(r, 0.004) + 2]
        r2 = r * r
        np.testing.assert_array_equal(r, local["r"][: len(r)])
        v = local["voltage"][: len(r)] - 300000.0
        knots = m["z"]
        gn = np.empty(2 * len(knots) - 1)
        gn[::2] = knots
        gn[1::2] = (knots[:-1] + knots[1:]) / 2
        verify = gn > knots[-1] - 0.040
        ind = m["electric_index"][verify]
        frac = m["electric_fraction"][verify]
        val = m["electric_v"][: len(r)] - 300000.0
        np.testing.assert_allclose(
            np.vstack([np.interp(gn[verify], zv, row) for row in v]),
            val[:, ind] + (val[:, ind + 1] - val[:, ind]) * frac,
            atol=3e-9,
            rtol=0,
        )
        j = m["electric_index"][0]
        start_phi = val[:, j] + (val[:, j + 1] - val[:, j]) * m["electric_fraction"][0]
        surface_phi = [np.interp(position * 0.001, zv, row) for row in v]
        energy += np.interp(
            (initial[:, :2] ** 2).sum(axis=1), r2, surface_phi
        ) - np.interp((gun[:, :2] ** 2).sum(axis=1), r2, start_phi)
        ix = {n: i for i, n in enumerate(d["selected_ids"])}
        previous = (d["z_mm"] >= 1534.5) & (d["z_mm"] < position - 2)
        zs = d["z_mm"][previous]
        traces = d["paths_mm"][[ix[k] for k in ids]][:, previous]
        rows = [
            (z, *widths_um(xy.astype(float) * 1000, weights))
            for z, xy in zip(zs, traces.swapaxes(0, 1))
        ]
    bounds = magnetic.x[[0, -1]]

    def follow(offsets):
        # Ordered offsets include zero first and grow monotonically in one direction.
        z = (position + offsets) * 0.001
        if len(z) == 1:
            return initial, initial[:, None, :2]
        if min(z) < max(zv[0], bounds[0]) or max(z) > min(zv[-1], bounds[-1]):
            raise ValueError("Focus outside the measured electric or magnetic volume")
        nodes = np.empty(2 * len(z) - 1)
        nodes[::2] = z
        nodes[1::2] = (z[1:] + z[:-1]) * 0.5
        i = np.searchsorted(zv, nodes, side="right") - 1
        delta = v[:, i + 1] - v[:, i]
        dz = zv[i + 1] - zv[i]
        slope = delta / dz
        phi = (v[:, i] + slope * (nodes - zv[i])).T
        ez = -slope.T
        ep = np.ascontiguousarray(
            np.stack(
                [
                    phi[:, :-1],
                    np.diff(phi) / np.diff(r2),
                    ez[:, :-1],
                    np.diff(ez) / np.diff(r2),
                ],
                axis=-1,
            )
        )
        end, loss, paths = trace_column(
            initial,
            energy,
            z,
            np.full(len(z) - 1, scales[-1]),
            np.arange(len(z)),
            magnetic(nodes),
            scales,
            ep,
            r2,
            factor,
        )
        if np.any(loss != -1):
            raise ValueError("A transmitted particle left the local focus volume")
        return end, paths

    rows.append((position, *widths_um(initial[:, :2] * 1e6, weights)))

    def record(offsets):
        _, paths = follow(offsets)
        rows.extend(
            (position + s, *widths_um(q.astype(float) * 1e6, weights))
            for s, q in zip(offsets[1:], paths[:, 1:].swapaxes(0, 1))
        )

    for direction, extent in ((-1, 2), (1, 1.6)):
        record(direction * np.arange(round(extent / 0.0025) + 1) * 0.0025)
    coarse = np.array(rows)
    ix = np.argmin(coarse[:, 1:], axis=0)
    if np.any(abs(coarse[ix, 0] - position) > 0.2):
        raise ValueError("Expand the fine focus search around the observed branch")
    # Both fine and coarse samplings use full Lorentz transport to the observed plane.
    for direction in (-1, 1):
        extent = max(
            0.010, float(np.max(direction * (coarse[ix, 0] - position))) + 0.0075
        )
        record(direction * np.arange(int(np.ceil(extent / 0.00025)) + 1) * 0.00025)
    result = np.array(sorted(rows, key=lambda row: row[0]))
    result = result[np.r_[True, np.diff(result[:, 0]) > 1e-10]]
    best = np.argmin(result[:, 1:], axis=0)
    states = {"surface": initial}
    summary = {}
    for j, key in enumerate(("D95", "rms", "x_rms", "y_rms")):
        z = result[best[j], 0]
        offset = z - position
        if best[j] in (0, len(result) - 1):
            raise ValueError("Unbracketed waist")
        end, _ = (
            follow(
                np.linspace(0, offset, max(2, 1 + int(np.ceil(abs(offset) / 0.00025))))
            )
            if abs(offset) > 1e-10
            else (initial, None)
        )
        if key in ("D95", "rms"):
            states[key] = end
        summary[key] = dict(
            z_mm=float(z),
            offset_um=float(offset * 1000),
            widths_um=widths_um(end[:, :2] * 1e6, weights),
        )
    np.savez_compressed(
        folder / "waists.npz",
        profiles=result,
        sample_ids=ids,
        weights_uA=weights,
        **states,
    )
    record = dict(
        specimen_z_mm=position,
        coarse_step_um=2.5,
        fine_step_um=0.25,
        planes=summary,
        extent_mm=result[[0, -1], 0].tolist(),
        surface_widths_um=widths_um(initial[:, :2] * 1e6, weights),
    )
    (folder / "waists.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


if __name__ == "__main__":
    import argparse
    from numba import get_num_threads, set_num_threads

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_folder", type=Path)
    args = parser.parse_args()
    set_num_threads(min(12, get_num_threads()))
    print(
        json.dumps(
            measure_waists(
                args.trace_folder, Path(__file__).with_name("specimen-electric.npz")
            ),
            indent=2,
        )
    )

"""Cross-check current closure, positive-weight prefixes, step size and the field on occupied rays."""

import argparse
import json
import numpy as np
from scipy.interpolate import CubicSpline
from relay import BASE, CACHE, HERE, WORK, Vacuum, groups, layered, load_bias, sha, values, wq, write
from volume import basis
from column_trace import trace


def run(folder):
    r = json.loads((folder / "result.json").read_text())
    path = WORK / (r["field"] + ".npz")
    table = layered(path)
    with np.load(folder / "trajectories.npz") as state:
        ids = state["ids"]
        top = state["surface_phase"]
        ok = state["loss"] == -1
        assert np.all(state["loss"] >= -1) and len(np.unique(ids)) == len(ids)
        w = state["weights_uA"]
        radius = np.maximum(
            np.linalg.norm(state["aperture_phase"][:, :2], axis=1),
            np.linalg.norm(state["aperture_rear_phase"][:, :2], axis=1),
        )
        selected = np.isin(state["selected_ids"], ids[ok])
        zz = state["z_mm"] * 0.001
        positions = state["xy_m"][:, selected]
        prefixes = []
        for n in (131072, 262144, 524288, 1048576):
            if n > r["emitted_particles"]:
                continue
            keep = ok & (ids < n)
            ww = w[keep] * r["emitted_particles"] / n
            p = top[keep]
            v = p[:, 2:4]
            chief = np.r_[np.average(v, weights=ww, axis=0), 1.0]
            dirs = np.column_stack((v, np.ones(len(v))))
            ang = np.arctan2(np.linalg.norm(np.cross(dirs, chief), axis=1), dirs @ chief) * 1000
            rim = radius[keep] >= 47.5e-6
            prefixes.append(
                dict(
                    n=n,
                    **values(p, ww),
                    rim_n=int(rim.sum()),
                    rim_median_mrad=wq(ang[rim], ww[rim], 0.5),
                )
            )
            if n == r["emitted_particles"]:
                members = []
                xy_relative = p[:, :2] - np.average(p[:, :2], axis=0, weights=ww)
                slope_relative = v - chief[:2]
                radial_products = np.einsum("ij,ij->i", xy_relative, slope_relative)
                for j in np.flatnonzero(rim):
                    members.append(
                        dict(
                            source_id=int(ids[keep][j]),
                            radius_um=float(radius[keep][j] * 1e6),
                            angle_mrad=float(ang[j]),
                            radially_inward=bool(radial_products[j] < 0),
                            sample_xy_um=(p[j, :2] * 1e6).tolist(),
                            represented_current_pA=float(ww[j] * 1e6),
                        )
                    )
                write(folder / "aperture-edge.json", sorted(members, key=lambda x: x["radius_um"]))
    rng = np.random.default_rng(9527)
    q = []
    for lo, hi in ((0.45, 0.77), (0.77, 1.32), (1.32, 1.5125), (1.5125, 1.59921)):
        zs = np.where((zz >= lo) & (zz < hi))[0]
        i = rng.choice(zs, 350)
        j = rng.choice(positions.shape[1], 350)
        q.extend(np.column_stack((positions[i, j], zz[i])))
    q = np.asarray(q)
    direct = Vacuum(path).symmetric(q)
    with np.load(table) as d:
        pred = np.zeros_like(direct)
        left = np.ones(len(q))
        rad = np.linalg.norm(q[:, :2], axis=1)
        sc = d["scales"]
        for i, scale in enumerate(sc):
            blend = np.clip((scale - rad) / (0.16 * scale), 0, 1) if i < len(sc) - 1 else np.ones(len(q))
            blend *= blend * (3 - 2 * blend)
            coef = CubicSpline(d["z_m"], d["layers"][i], axis=0)(q[:, 2])
            pred += (
                np.einsum("ij,ijk->ik", basis(q[:, :2] / scale, int(d["degree"])), coef)
                * (blend * left)[:, None]
            )
            left *= 1 - blend
    err = np.linalg.norm(pred - direct, axis=1).reshape(4, 350)
    spot = np.linalg.norm(direct, axis=1).reshape(4, 350)
    gunfile = CACHE / r["gun_packet"]
    with np.load(gunfile) as f:
        assert np.all(f["weight_uA"] > 0) and abs(f["weight_uA"].sum() - 15) < 1e-9
        gun_total = float(f["weight_uA"][f["label"] == 0].sum())
    ledger = r["current_ledger_A"]
    np.testing.assert_allclose(gun_total * 1e-6, ledger["gun_exit_A"], rtol=1e-12)
    np.testing.assert_allclose(w[ok].sum() * 1e-6, ledger["specimen_A"], rtol=1e-12)
    coarse = trace(path, table, gunfile, folder, 0.12, detail=False)
    assert coarse["surface"]["count"] == r["surface"]["count"]
    b0 = load_bias(BASE)
    bf = load_bias(path)
    changes = {k: [b0[k], v] for k, v in bf.items() if b0[k] != v}
    assert set(changes) <= set(groups(bf)["objective"] + groups(bf)["tl"])
    res = dict(
        emission_uA=15.0,
        gun_exit_uA=gun_total,
        no_numerical_losses=True,
        source_prefixes=prefixes,
        sample_step_comparison={"0.06_mm": r["surface"], "0.12_mm": coarse["surface"]},
        changed_windings=changes,
        table_sha256=sha(table),
        field_sha256=sha(path),
        direct_material_field_spot_checks=dict(
            samples_each=350,
            intervals_mm=[[450, 770], [770, 1320], [1320, 1512.5], [1512.5, 1599.21]],
            median_p95_max_error_T=np.quantile(err, [0.5, 0.95, 1], axis=1).T.tolist(),
            peak_field_T=np.max(spot, axis=1).tolist(),
        ),
    )
    write(folder / "checks.json", res)
    print(
        "Verified source, step, current, and field",
        prefixes[-1],
        res["direct_material_field_spot_checks"],
        flush=True,
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("folder")
    run(HERE / a.parse_args().folder)

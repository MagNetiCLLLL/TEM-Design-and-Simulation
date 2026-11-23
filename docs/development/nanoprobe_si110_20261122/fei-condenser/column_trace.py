"""Transport actual gun-exit IDs to every physical stop and to the specimen surface."""

import argparse
import json

import cupy as cp
import numpy as np
from scipy.interpolate import CubicSpline

from gun_images import HERE
from relaunch import relaunch
from volume import layered, sample, solution
from condenser_pair_column import column_mesh
from material_relay_column import source_and_electrodes
from column_current import current_ledger
from particles import Tracker, HERE as NATIVE
from relay_transfer import advance, electric_reference
from search_pair import FACTOR as FACTOR_LOCAL
from transport import REST_EV


def source_packet(gunfile):
    with np.load(gunfile) as d:
        live = d["label"] == 0
        u = d["end"][live]
        w = d["weight_uA"][live]
        assert len(d["ids"]) >= 1000 and np.all(d["weight_uA"] > 0)
        phase = np.column_stack((u[:, :2], u[:, 3:5] / u[:, 5, None], u[:, 6]))
        u2 = (u[:, 3:6] ** 2).sum(axis=1)
        energy = REST_EV * u2 / (np.sqrt(1 + u2) + 1)
        return phase, w / 15.0, d["ids"][live], energy - 300000.0, 15e-6


def optics_mesh(step):
    # Join the FEI outlet and the installed downstream electrodes inside
    # their common grounded tube. Retain the potential near the specimen.
    from search_pair import GUN_PACKET

    f = source_and_electrodes()
    m, stops = column_mesh(
        f, step, extra_observation_mm=(764.9, 765.1, 1009.9, 1010.1, 1179.1, 1237.2, 1420.0, 1512.5)
    )
    old = f.electric.base_field
    join = 0.490
    with np.load(GUN_PACKET.parent / "anode-mesh15-50V.npz") as d:
        rr = d["r"]
        zz = d["z"]
        fg = d["voltage"]
        r = old.r[: np.searchsorted(old.r, 0.0042) + 1]
        head = np.flatnonzero((zz >= 0.448) & (zz < join))
        tail = np.flatnonzero((old.z > join) & (old.z < 1.611))
        mg = np.column_stack([np.interp(r * r, rr * rr, fg[:, i]) for i in head])
        fg_join = np.array([np.interp(join, zz, row) for row in fg])
        old_join = np.array([np.interp(join, old.z, row) for row in old.voltage[: len(r)]])
        supplied = np.interp(r * r, rr * rr, fg_join)
        old_deviation = float(np.max(abs(supplied - old_join)))
        if old_deviation > 0.001:
            raise ValueError(f"Potential mismatch at 490 mm: {old_deviation} V")
        z = np.r_[zz[head], join, old.z[tail]]
        volts = np.column_stack((mg, supplied, old.voltage[: len(r), tail]))
    j = np.clip(np.searchsorted(z, m.nodes, side="right") - 1, 0, len(z) - 2)
    dz = np.diff(z)[j]
    m.electric = (r, volts, j, (m.nodes - z[j]) / dz, dz)
    return m, stops, float(old_deviation)


class BoreTracker(Tracker):
    def __init__(self, mesh, table, source):
        super().__init__(mesh, source=source)
        with np.load(table) as d:
            degree = int(d["degree"])
            scale = float(d["scale_m"])
            if "layers" in d:
                scales = d["scales"]
                coeff = CubicSpline(d["z_m"], d["layers"].transpose(1, 0, 2, 3), axis=0)(mesh.nodes)
            else:
                scales = np.array([scale])
                coeff = CubicSpline(d["z_m"], d["coeff"], axis=0)(mesh.nodes)
        self.tables = (cp.asarray(np.ascontiguousarray(coeff)),)
        native = (NATIVE / "lorentz.cu").read_text()
        scale_literal = ",".join(f"{v:.17g}" for v in scales)
        header = (
            f"#define DEG {degree}\n#define SCALE {scale:.17g}\n#define NLAYER {len(scales)}\n"
            + f"#define SCALE_ARRAY {{{scale_literal}}}\n"
            + "#define CORRECTOR_FIELD_PARAMETERS , const double* coeff\n"
            + "#define CORRECTOR_FIELD_ARGUMENTS ,coeff\n"
        )
        hook = "int lo=0,hi=nr-1;"
        poly = """
        const int nterm=(DEG+1)*(DEG+2)/2;
        const double scales[NLAYER]=SCALE_ARRAY;
        const double radxy=sqrt(s);
        double remaining=1.;
        for(int l=0;l<NLAYER;l++){
            double blend=1.;
            if(l<NLAYER-1){
                double t=fmin(1.,fmax(0.,(scales[l]-radxy)/(.16*scales[l])));
                blend=t*t*(3.-2.*t);
            }
            double weight=remaining*blend;remaining-=weight;
            if(weight<=0.)continue;
            const double *cc=coeff+((size_t)node*NLAYER+l)*nterm*3;
            double xp[DEG+1],yp[DEG+1];xp[0]=yp[0]=1.;
            for(int k=1;k<=DEG;k++){xp[k]=xp[k-1]*x/scales[l];yp[k]=yp[k-1]*y/scales[l];}
            int term=0;
            for(int d=0;d<=DEG;d++)for(int i=0;i<=d;i++,term++){
                double t=weight*xp[i]*yp[d-i];bx+=t*cc[3*term];by+=t*cc[3*term+1];bz+=t*cc[3*term+2];
            }
            if(remaining<=0.)break;
        }
        """
        contact = "if(endpoints>radius2[i]) {loss[j]=i;break;}"
        if native.count(hook) != 1 or native.count(contact) != 1:
            raise ValueError("Changed transport contract")
        native = native.replace(hook, poly + hook).replace(
            contact,
            "if(endpoints>radius2[i] && radius2[i]<=SCALE*SCALE) {loss[j]=i;break;}\n"
            + "if(swept>SCALE*SCALE){loss[j]=-3-i;break;}\n"
            + contact,
        )
        self.module = cp.RawModule(code=header + native, options=("--std=c++11", "--fmad=false"))
        self._trace_kernel = self.module.get_function("trace")


def wq(a, w, p=0.95):
    order = np.argsort(a)
    acc = np.cumsum(w[order])
    return float(a[order[np.searchsorted(acc, p * acc[-1])]])


def values(p, w):
    centre = np.average(p[:, :2], weights=w, axis=0)
    mean = np.average(p[:, 2:4], weights=w, axis=0)
    u = p[:, :2] - centre
    chief = np.r_[mean, 1.0]
    chief /= np.linalg.norm(chief)
    unit = np.column_stack((p[:, 2:4], np.ones(len(p))))
    angle = np.arctan2(np.linalg.norm(np.cross(unit, chief), axis=1), unit @ chief)
    rad = np.linalg.norm(u, axis=1)
    return dict(
        D95_um=2e6 * wq(rad, w),
        rms_um=1e6 * np.sqrt(np.average(rad**2, weights=w)),
        alpha95_mrad=1e3 * wq(angle, w),
        axis_alpha95_mrad=1e3 * wq(np.arctan(np.linalg.norm(p[:, 2:4], axis=1)), w),
        axis_core_50pm_pA=1e6 * w[np.linalg.norm(p[:, :2], axis=1) <= 50e-12].sum(),
        count=len(p),
        ess=w.sum() ** 2 / (w @ w),
        sample_nA=1e3 * w.sum(),
        centroid_um=(1e6 * centre).tolist(),
    )


def covariance(p, w):
    xy = p[:, :2]
    v = p[:, 2:4]
    xy = xy - np.average(xy, weights=w, axis=0)
    v = v - np.average(v, weights=w, axis=0)
    rms = np.average(xy**2, weights=w, axis=0)
    cov = np.average(xy * v, weights=w, axis=0)
    return np.r_[rms, cov]


def crossings(z, cov):
    i = np.where((cov[:-1] < 0) & (cov[1:] >= 0))[0]
    return (z[i] - cov[i] * np.diff(z)[i] / (cov[i + 1] - cov[i])).tolist()


def propagate(t, m, stops):
    n = len(t.ids)
    p = t.input.copy()
    lost = np.full(n, -1, np.int32)
    first = 0
    blank = cp.zeros((len(m.nodes), 20))
    energy = 300000.0 + t.de
    phi0 = electric_reference(t)
    for group in np.array_split(stops, 1 + (len(stops) - 1) // 12):
        try:
            states = advance(t, blank, p, energy, phi0, lost, first, group)
        except ValueError:
            dev = cp.asarray(np.ascontiguousarray(p))
            hits = cp.array(lost)
            peak = cp.zeros(n)
            margin = cp.full(n, cp.inf)
            for j in range(first, group[-1], 160):
                t.kernel(
                    ((n + 127) // 128,),
                    (128,),
                    (
                        dev,
                        cp.asarray(energy),
                        cp.asarray(phi0),
                        blank,
                        t.ep,
                        t.er,
                        np.int32(len(t.r2)),
                        t.h,
                        t.rad,
                        hits,
                        peak,
                        margin,
                        np.int32(j),
                        np.int32(min(group[-1], j + 160)),
                        np.int32(n),
                        t.rest,
                        FACTOR_LOCAL,
                        t.pc,
                    ),
                )
            code = hits.get()
            bad = code < -2
            print(
                "Out-of-table original IDs, z, physical radius and RK-stage radius (mm)",
                np.column_stack(
                    (
                        t.ids[bad],
                        m.z[-3 - code[bad]] * 1000,
                        m.radius[-3 - code[bad]] * 1000,
                        np.sqrt(peak.get()[bad]) * 1000,
                    )
                )[:20],
                flush=True,
            )
            raise
        for k, (p, lost) in zip(group, states, strict=True):
            yield k, p, lost
        first = group[-1]


def trace(path, table, gunfile, tag, step, detail=True):
    source = source_packet(gunfile)
    m, stops, edev = optics_mesh(step)
    w = source[1] * 15
    t = BoreTracker(m, table, source)
    surface_index = len(m.z) - 1
    foil_idx = np.searchsorted(m.z, 0.7649)
    (_, foil, _), (_, rear, _), (_, top, loss) = list(
        propagate(t, m, [foil_idx, np.searchsorted(m.z, 0.7651), surface_index])
    )
    live = loss == -1
    if not np.any(live):
        raise ValueError("Empty surface")
    ledger = current_ledger(m, t, loss)
    surf = values(top[live], w[live])
    rim = np.maximum(np.linalg.norm(foil[live, :2], axis=1), np.linalg.norm(rear[live, :2], axis=1))
    vel = top[live, 2:4]
    chief = np.r_[np.average(vel, weights=w[live], axis=0), 1.0]
    chief /= np.linalg.norm(chief)
    dirs = np.column_stack((vel, np.ones(len(vel))))
    ang = 1000 * np.arctan2(np.linalg.norm(np.cross(dirs, chief), axis=1), dirs @ chief)
    band = rim >= 0.95 * 50e-6
    at_surface = covariance(top[live], w[live])
    result = dict(
        c1_A=float(tables_bias(path, 0)),
        c2_A=float(tables_bias(path, 1)),
        field=path.stem,
        field_table=table.name,
        gun_packet=gunfile.name,
        step_mm=step,
        emission_uA=15.0,
        potential_join_max_difference_V=edev,
        surface=surf,
        current_ledger_A=ledger,
        surface_rms_radius_slope_mrad=float(1000 * at_surface[2:].sum() / np.sqrt(at_surface[:2].sum())),
        rim_outer_5pct=dict(
            count=int(band.sum()),
            max_radius_um=float(rim.max() * 1e6),
            angle_range_mrad=[float(ang[band].min()), float(ang[band].max())] if band.any() else [],
            median_mrad=wq(ang[band], w[live][band], 0.5) if band.any() else None,
            percentile95_mrad=wq(ang[band], w[live][band]) if band.any() else None,
        ),
    )
    if detail:
        folder = HERE / tag
        folder.mkdir(exist_ok=True)
        ids = np.flatnonzero(live)
        other = np.flatnonzero(~live)
        # Stratify the picture by incoming radius; moments and ledgers use all IDs.
        ids = ids[np.argsort(rim)]
        take = ids[np.linspace(0, len(ids) - 1, min(70, len(ids)), dtype=int)]
        if len(other):
            take = np.r_[take, other[np.linspace(0, len(other) - 1, min(28, len(other)), dtype=int)]]
        zs = []
        stats = []
        fixed = []
        paths = []
        focus = []
        for k, p, lost in propagate(t, m, stops):
            z = m.z[k] * 1000
            alive = lost == -1
            stats.append(covariance(p[alive], w[alive]))
            fixed.append(covariance(p[live], w[live]))
            paths.append(np.where(alive[take, None], p[take, :2], np.nan))
            zs.append(z)
            if z >= 1512.5:
                focus.append((z, values(p[live], w[live])["D95_um"]))
        zs = np.asarray(zs)
        stats = np.array(stats)
        fixed = np.array(fixed)
        result["waists"] = {}
        for name, a in [("live", stats), ("same_specimen_ids", fixed)]:
            result["waists"][name] = dict(
                radial_mm=crossings(zs, a[:, 2:].sum(axis=1)),
                x_mm=crossings(zs, a[:, 2]),
                y_mm=crossings(zs, a[:, 3]),
            )
        result["focus_upstream_min_mm_D95um"] = min(focus, key=lambda x: x[1])
        np.savez_compressed(
            folder / "trajectories.npz",
            z_mm=zs,
            stats_m=stats,
            fixed_stats_m=fixed,
            xy_m=paths,
            selected_ids=t.ids[take],
            weights_uA=w,
            ids=t.ids,
            surface_phase=top,
            loss=loss,
            aperture_phase=foil,
            aperture_rear_phase=rear,
            focus_upstream=focus,
            gun_exit_phase=t.input,
        )
        (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(tag, json.dumps(result), flush=True)
    return result


def tables_bias(path, i):
    with np.load(path) as d:
        b = json.loads(str(d["report"]))["excitation"]
    return b[["condenser_lens_1/physical-pack-0", "condenser_lens_2/physical-pack-1"][i]]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("c1", type=float)
    p.add_argument("c2", type=float)
    p.add_argument("tag")
    p.add_argument("--count", type=int, default=131072)
    p.add_argument("--step", type=float, default=0.12)
    p.add_argument("--fine", action="store_true")
    p.add_argument("--nested", action="store_true")
    a = p.parse_args()
    path = solution(a.c1, a.c2)
    trace(
        path,
        layered(path, a.fine) if a.nested else sample(path, False, a.fine),
        relaunch(path, a.count),
        a.tag,
        a.step,
    )

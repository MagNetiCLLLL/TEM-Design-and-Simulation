"""Adjust the two condenser supplies against the installed inter-lens gaps."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import cupy as cp
import numpy as np
from numba import njit, prange, set_num_threads
from scipy.stats import qmc

from column_current import ACTIVE, FACTOR, OWNER, POINT, current_ledger, fields, metrics, source_digest
from column import read_gun
from observe import axis_crossings
from relay_transfer import FOIL, HERE, HP1, HP2, RADIUS, START, SURFACE, WORK, Native, mechanical_span_mm


FIELD = WORK / "m330-field.npz"
PUPIL = 1.40805425036
LENSES = ("condenser_lens_2", "condenser_lens_3", "adapter_lens")
CASES = (
    ("before", 1., 1.), ("c2-only", 1.542, 1.), ("weak-only", 1., .26),
    ("uniform", 1.542, .26), ("g2-1548", 1.548, .26), ("g3-024", 1.542, .24),
    ("wider-footprint", 1.472, .185), ("larger-angle", 1.846, .335),
)


def reference():
    seed = json.loads((HERE / "evidence/internal-optics.json").read_text(encoding="utf-8"))["candidate"]
    if seed["run"] != "internal-waist-1171-0-2" or sha256(FIELD.read_bytes()).hexdigest() != seed["field_sha256"]:
        raise ValueError("The previous working point and its registered field are required")
    return seed


def spans_mm():
    return np.array([mechanical_span_mm(name) for name in LENSES])


def coils_at(seed, g2, g3):
    q = np.array(seed["currents_A"])
    for gain, owner in zip((g2, g3), LENSES[:2]):
        q[OWNER[ACTIVE] == owner] *= gain
    return q


@njit(cache=True, parallel=True)
def prescreen(gains, profile, h, z, events, bands, initial, radius_max):
    result = np.zeros((len(gains), 10))
    for j in prange(len(gains)):
        k0 = FACTOR * (profile[0, 0] + np.dot(profile[0, 1:], gains[j]))
        w, v = 1.0 + 0j, initial - 1j * k0
        rf = 0.0
        for i in range(len(h)):
            bz = profile[i+1, 0]
            for group in range(gains.shape[1]):
                bz += profile[i+1, group+1] * gains[j, group]
            k = FACTOR * bz
            kh = k * h[i]
            co = np.cos(kh)
            si = np.sin(kh) / k if abs(k) > 1e-13 else h[i]
            wo, vo = w, v
            w, v = co * wo + si * vo, -k*k*si*wo + co*vo
            if (wo.conjugate() * vo).real < 0 <= (w.conjugate() * v).real:
                if bands[0] < z[i] < bands[1]:
                    result[j, 0] += 1
                    result[j, 2] = z[i] * 1000
                if bands[2] < z[i] < bands[3]:
                    result[j, 1] += 1
                if bands[4] < z[i] < bands[5]:
                    result[j, 6] += 1
                if bands[5] < z[i] < bands[6]:
                    result[j, 7] += 1
                if bands[7] < z[i] < bands[8]:
                    result[j, 8] += 1
            if i+1 == events[0] or i+1 == events[1]:
                rf = max(abs(w), rf)
        ke = FACTOR * (profile[-1, 0] + np.dot(profile[-1, 1:], gains[j]))
        slope = v + 1j*ke*w
        result[j, 3] = 1000*np.arctan(RADIUS*abs(slope)/rf)
        result[j, 4] = RADIUS*abs(w)/rf*1e6
        result[j, 5] = radius_max*rf/RADIUS
        result[j, 9] = (w.conjugate()*slope).real
    return result


def scan_inputs(split, step, *, illuminate=False):
    seed = reference()
    q = np.array(seed["currents_A"])
    faces = spans_mm() * .001
    c2, c3, adl = faces.mean(axis=1)
    m, table, bg, coils = fields(step, np.r_[faces.ravel(), FOIL, SURFACE], seed["objective_At"], FIELD)
    center = len(table["xy_m"]) // 2
    axis = np.interp(m.nodes - START, table["z_local_m"], table["B_zyx_T"][:, center, center, 2])
    names = ("condenser_lens_1", *LENSES[:2]) if illuminate else LENSES[:2]
    grouped = [coils[OWNER[ACTIVE] == key, :, 0].T @ q[OWNER[ACTIVE] == key] for key in names]
    fixed = bg[:, 0] + axis + coils[:, :, 0].T @ q - sum(grouped)
    if split:
        grouped = [*coils[OWNER[ACTIVE] == LENSES[0], :, 0], grouped[1]]
    profile = np.ascontiguousarray(np.column_stack([a[np.r_[0, 1:len(a):2, len(a)-1]] for a in (fixed, *grouped)]))
    initial, weights, *_ = read_gun(POINT)
    xy, tx = initial[:, 0] + 1j*initial[:, 1], initial[:, 2] + 1j*initial[:, 3]
    slope = np.vdot(weights * xy, tx) / np.vdot(weights * xy, xy)
    bands = np.array([faces[0, 1], faces[1, 0], faces[1, 1], faces[2, 0], c2, c3, adl, 1.5225, SURFACE-.00025])
    set_num_threads(10)
    return profile, np.diff(m.z), m.nodes[1::2], np.searchsorted(m.z, FOIL), bands, slope, np.max(abs(xy))


def scan(split=False):
    inputs = scan_inputs(split, .40 if split else .30)
    if split:
        # Preserve the four winding polarities and do not exceed the largest
        # previous C2 current. All other changes are a single common C3 gain.
        seed = reference()
        section = np.array(seed["currents_A"])[OWNER[ACTIVE] == LENSES[0]]
        trial = qmc.Sobol(5, scramble=True, seed=20261121).random_base2(17)
        factors = np.column_stack((trial[:,:4]*np.sign(section)*max(abs(section)), .06+.34*trial[:,4]))
        result = prescreen(factors,*inputs)
        np.savez_compressed(WORK/"normal-c2-sections.npz", gains=factors, result=result, spans_mm=spans_mm())
        ok = (result[:,0]==1)&(result[:,1]==0)&(result[:,6]==1)&(result[:,7]==0)&(result[:,8]==0)
        print('Valid branch / illuminated rim',np.count_nonzero(ok),np.count_nonzero(ok&(result[:,5]>1.1)),flush=True)
        ix = np.flatnonzero(ok)
        if len(ix):
            for fill in (.9,1.1,1.5):
                valid = ix[result[ix,5]>fill]
                if len(valid):
                    k = valid[np.argmin(abs(result[valid,3]-30)+.1*result[valid,4])]
                    print('fill',fill,'windings & C3',factors[k], 'waist/angle/radius/fill',result[k,2:6],flush=True)
        return
    factors = np.stack(np.meshgrid(np.arange(-2.5, 2.501, .002), np.unique(np.r_[np.arange(.05, .801, .015), 1.])), axis=-1).reshape(-1, 2)
    result = prescreen(factors,*inputs)
    np.savez_compressed(WORK / "normal-condenser-screen.npz", gains=factors, result=result, spans_mm=spans_mm())
    good = (result[:, 0] == 1) & (result[:, 1] == 0) & (result[:, 6] == 1) & (result[:, 7] == 0)
    for g3 in (1., .80, .65, .5, .35, .20, .05):
        mask = good & np.isclose(factors[:, 1], g3)
        idx = np.flatnonzero(mask)
        if len(idx):
            best = idx[np.argmin(abs(result[idx, 3]-30))]
            print('C3 gain',g3,'C2 bands',factors[idx[[0,-1]],0],'angle range',np.round([result[idx,3].min(),result[idx,3].max()],2),
                  'closest: gains / waist mm, edge mrad, marginal r um, foil fill, center counts, last waist count',
                  factors[best],np.round(result[best,2:9],3),flush=True)
        else:
            print('C3 gain',g3,'no candidates',flush=True)


def physical_observation(model, q, obs, name):
    field, (front, rear, (last, loss)) = model.replay(q, [*FOIL, SURFACE])
    t = model.t
    ok = loss == -1
    # Keep the *same* original source IDs in every display. Their actual foil
    # radii and loss states can change when a lens current changes.
    with np.load(HERE / "evidence/last-point-trajectories.npz") as reference:
        display_ids = reference["source_id"]
        present = np.isin(display_ids, t.ids)
        lookup = {source_id: i for i, source_id in enumerate(t.ids)}
        ids = np.array([lookup[source_id] for source_id in display_ids[present]])
        bands = reference["band"][present]
    fill = np.linalg.norm(front[0][:,:2], axis=1)/RADIUS
    weights = t.weights[ok]
    specimen = metrics(last[ok], weights, t.current)
    footprint = np.maximum.reduce([np.linalg.norm(a[0][:, :2], axis=1) / RADIUS for a in (front, rear)])
    incident = front[1] == -1
    radius = np.hypot(*front[0][incident, :2].T)
    stats = dict(specimen=specimen, source_count=len(ok), survivors=int(ok.sum()), effective=float(weights.sum()**2 / (weights@weights)),
                 source_digest=source_digest(t), integration_step_mm=model.step, currents_A=list(q), objective_At=model.ni,
                 max_incident_foil_radius_um=float(max(radius)*1e6), max_passing_foil_radius_um=float(max(footprint[ok])*RADIUS*1e6),
                 foil_rim_count=int(np.count_nonzero(ok & (footprint >= .90))),
                 max_source_angle_mrad=float(1e3*np.max(np.arctan(np.linalg.norm(last[ok, 2:4],axis=1)))),
                 current_ledger=current_ledger(model.m, t, loss))
    if stats["foil_rim_count"] > 0:
        try:
            data = model.entrance(field, front)
            stats["edge"] = model.edge(field, data, azimuths=128)
            stats["rim_fit"] = data[1]
        except ValueError as err:
            stats["edge_unresolved"] = str(err)
    p, en, phi = cp.asarray(t.input).copy(), cp.asarray(model.en), cp.asarray(model.phi0)
    n = len(ok)
    lost, peak, margin = cp.full(n, -1, np.int32), cp.zeros(n), cp.full(n, cp.inf)
    survivor_indices = cp.asarray(np.flatnonzero(ok))
    w = cp.asarray(weights / weights.sum())
    selected = cp.asarray(ids)
    paths = cp.empty((len(obs), len(ids), 4))
    moments = cp.empty((len(obs), 5))
    start = 0
    for j, end in enumerate(model.indices(obs)):
        for i in range(start, end, 160):
            t.kernel(((n+127)//128,), (128,), (p, en, phi, field, t.ep, t.er, np.int32(len(t.r2)), t.h, t.rad,
                lost, peak, margin, np.int32(i), np.int32(min(i+160,end)), np.int32(n), t.rest, FACTOR, t.pc))
        paths[j] = cp.where((lost[selected] == -1)[:, None], p[selected,:4], cp.nan)
        phase = p[survivor_indices,:4]
        centered = phase - cp.sum(phase*w[:,None],axis=0)
        moments[j,0] = weights.sum()
        moments[j,1:3] = cp.sqrt(cp.sum(w[:,None]*centered[:,:2]**2,axis=0))
        moments[j,3:] = cp.sum(w[:,None]*centered[:,:2]*centered[:,2:],axis=0)
        start = end
    np.testing.assert_array_equal(p.get(), last)
    np.testing.assert_array_equal(lost.get(), loss)
    mom = moments.get()
    faces = spans_mm()*.001
    bounds = {"C2_body": faces[0], "C2_C3_gap": [faces[0,1], faces[1,0]], "C3_body": faces[1],
              "C3_ADL_gap": [faces[1,1],faces[2,0]], "C3_center_ADL_center": [faces[1].mean(), faces[2].mean()],
              "C2_center_C3_center": [faces[0].mean(), faces[1].mean()],
              "mini_to_presample": [1.5225,SURFACE-.00025]}
    stats["x_y_waists_mm"] = {k:axis_crossings(obs,mom,*limits) for k,limits in bounds.items()}
    stats["lens_fields"] = {}
    for key,face in zip(LENSES[:2],faces):
        mask = OWNER[ACTIVE] == key
        bz = model.coils[mask,:,0].T @ q[mask]
        z = model.m.nodes
        region=(z>=face[0])&(z<=face[1])
        stats["lens_fields"][key] = dict(peak_self_T=float(max(abs(bz[region]))),
            integral_self_B2_T2m=float(np.trapezoid(bz*bz,z)))
    output = WORK / (name + f"-{n}-{model.step:g}")
    (output.parent / (output.name + ".json")).write_text(json.dumps(stats,indent=2,allow_nan=False),encoding="utf-8")
    np.savez_compressed(output.parent / (output.name + ".npz"), z_m=obs, moments=mom, paths=paths.get(),
                        path_ids=t.ids[ids], path_band=bands, path_C2_fill=fill[ids], source_id=t.ids, weights=t.weights,
                        energy_offset_eV=t.de, phase=last, loss=loss)
    return stats


def observation_planes(spacing):
    faces = spans_mm()
    obs_mm = np.unique(np.r_[START*1000,np.arange(450.,552.,3.),
                            np.arange(550.,1090.,spacing),faces.ravel(),faces.mean(axis=1),
                            FOIL*1000, np.arange(1090.,1598.,1.),[1144.5,1166.35,1249.95,1300.85,1522.5],
                            np.arange(SURFACE*1000-.32,SURFACE*1000,.005),SURFACE*1000])
    return np.unique(np.r_[obs_mm[obs_mm < SURFACE*1000]*.001, FOIL, SURFACE])


def trace(args):
    seed = reference()
    obs = observation_planes(args.spacing)
    model = Native(FIELD, args.step, args.count, seed["objective_At"], np.r_[obs, HP2, HP1, PUPIL])
    cases = (json.loads(args.cases.read_text(encoding="utf-8")) if args.cases else
             [dict(name="condenser-" + name, C2=g2, C3=g3) for name, g2, g3 in CASES])
    for row in cases:
        q = coils_at(seed, row["C2"], row["C3"])
        if "C1_illumination" in row:
            q[OWNER[ACTIVE] == "condenser_lens_1"] *= row["C1_illumination"]
        if "C2_windings_A" in row:
            q[OWNER[ACTIVE] == LENSES[0]] = row["C2_windings_A"]
        if "mini_gain" in row:
            q[OWNER[ACTIVE] == "mini_condenser"] *= row["mini_gain"]
        s = physical_observation(model, q, obs, row["name"])
        print(row["name"], "gap waists", s['x_y_waists_mm']['C2_C3_gap'], s['x_y_waists_mm']['C3_ADL_gap'],
              "foil radius um / rim count", round(s["max_incident_foil_radius_um"],3),s["foil_rim_count"],
              "edge mrad",s.get("edge",{}).get("angle_mrad_min_median_max"),"D95 um",s["specimen"]["D95_nm"]*.001,
              "max source mrad",s["max_source_angle_mrad"],flush=True)


def check_waist(args):
    """Replay both sides of the measured waist and test all transverse directions."""
    seed = reference()
    obs = observation_planes(args.spacing)
    model = Native(FIELD, args.step, args.count, seed["objective_At"], np.r_[obs, HP2, HP1, PUPIL])
    q = coils_at(seed, 1.542, .26)
    record = json.loads((WORK / f"condenser-uniform-{args.count}-{args.step:g}.json").read_text())
    if record['source_digest'] != source_digest(model.t) or not np.array_equal(record['currents_A'], q):
        raise ValueError("The waist test requires the same source IDs, weights and drives")
    roots = np.asarray(record['x_y_waists_mm']['C2_C3_gap'])
    if roots.shape != (2, 1, 2):
        raise ValueError("Expected one upstream waist bracket in each coordinate")
    left, right = roots.min(), roots.max()
    near = obs[(obs * 1000 >= left - 1.01) & (obs * 1000 <= right + 1.01)]
    _, states = model.replay(q, [*near, SURFACE])
    live = states[-1][1] == -1
    w = model.t.weights[live]
    w = w / w.sum()
    radii, derivatives = [], []
    for phase, lost in states[:-1]:
        if np.any(lost[live] != -1):
            raise ValueError("A source particle reappeared after interception")
        phase = phase[live, :4]
        phase = phase - w @ phase
        pos, slope = phase[:, :2], phase[:, 2:]
        cross = pos.T @ (w[:, None] * slope)
        radii.append(pos.T @ (w[:, None] * pos))
        derivatives.append(np.linalg.eigvalsh(cross + cross.T))
    derivative = np.asarray(derivatives)
    covariance = np.asarray(radii)
    mid = np.argmin(np.trace(covariance, axis1=1, axis2=2))
    growth = np.linalg.eigvalsh(covariance[[0, -1]] - covariance[mid])
    verified = (0 < mid < len(near)-1 and np.all(derivative[0] < 0) and
                np.all(derivative[-1] > 0) and np.all(growth > 0))
    output = dict(source_count=args.count, survivors=int(live.sum()), source_digest=source_digest(model.t),
                  z_mm=(near * 1000).tolist(), covariance_m2=covariance.tolist(),
                  covariance_derivative_eigenvalues_m=derivative.tolist(),
                  minimum_trace_z_mm=float(near[mid] * 1000),
                  endpoint_growth_eigenvalues_m2=growth.tolist(), contracts_and_expands_in_all_directions=bool(verified))
    (WORK / 'condenser-uniform-waist.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print('Two-dimensional covariance reversal:', verified, 'at', output['minimum_trace_z_mm'], 'mm')
    if not verified:
        raise ValueError("A two-dimensional waist was not verified")


def through_focus(args):
    from finite_probe import axial
    seed = reference()
    q = coils_at(seed, 1.542, .26)
    model = Native(FIELD, args.step, args.count, seed['objective_At'])
    name = 'condenser-uniform-axial'
    result = axial(q, model.ni, args.count, args.step, 15e-6, name, model.m.intervals,
                   FIELD, WORK, transverse_interpolation='cubic')
    (WORK / (name + '.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["screen", "sections", "trace", "waist", "focus"], default="screen", nargs="?")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--step", type=float, default=.12, help="maximum integration step in mm")
    parser.add_argument("--spacing", type=float, default=1., help="axial observation spacing in mm")
    args = parser.parse_args()
    if args.action in ("screen", "sections"):
        scan(args.action == "sections")
    elif args.action == "waist":
        check_waist(args)
    elif args.action == "focus":
        through_focus(args)
    else:
        trace(args)

"""Map the mechanical and magnetic reach of each illumination element at the reported bias."""

import json
from hashlib import sha256

import numpy as np
from scipy.interpolate import make_interp_spline

from column_current import ACTIVE, OWNER
from downstream_pivot import write
from illuminated_focus import column_planes
from material_field_ranges import CACHE
from relay_transfer import HERE, PARTS, START, SURFACE, WORK, Native
from temsim.corrector_field_cli import _save
from temsim.multipole_winding import multipole_winding_definition
from temsim.physics.corrector_vacuum_field import _induced_cuda
from temsim.physics.transverse_coil_field import _unit_nodes
from temsim.transverse_winding import transverse_winding_definition


EVIDENCE = HERE/"evidence"
ROUND = {"C1": "condenser_lens_1", "C2": "condenser_lens_2", "C3": "condenser_lens_3",
         "ADL": "adapter_lens", "TL12": "probe_tl12_lens", "TL11 / Mini": "mini_condenser"}
MATERIAL = {"HP2": "probe_hp2_hexapole", "TL22": "probe_tl22_lens", "TL21": "probe_tl21_lens",
            "HP1": "probe_hp1_hexapole", "HPol X": "probe_hpol_hexapole_x", "HPol Y": "probe_hpol_hexapole_y"}
STEERING = {"Condenser deflector upper": ("condenser_deflector", 0),
            "Condenser deflector lower": ("condenser_deflector", 1),
            "Beam deflector upper": ("beam_deflector", 0), "Beam deflector lower": ("beam_deflector", 1),
            "DPH2": ("probe_dph2_coils", 0), "DP22": ("probe_dp22_coils", 0),
            "DP21": ("probe_dp21_coils", 0), "DPH1": ("probe_dph1_coils", 0),
            "DP11": ("probe_dp11_coils", 0), "Upper scan": ("ac_deflector", 0),
            "Lower scan": ("ac_deflector", 1)}
TRIMS = {"QPH2": "probe_qph2_quadrupole", "HPC": "probe_hpc_hexapole",
         "QPC": "probe_qpc_quadrupole", "QPH1": "probe_qph1_quadrupole",
         "QPol": "probe_qpol_quadrupole", "Condenser stigmator": "condenser_stigmator"}


def bands(z, values, fraction):
    amplitude = np.abs(values)
    level = amplitude.max()*fraction
    if level <= 0 or not np.isfinite(amplitude).all():
        raise ValueError("Cannot normalize a zero or invalid channel")
    high = amplitude >= level
    j = np.flatnonzero(high[1:] != high[:-1])
    crosses = z[j] + (z[j+1]-z[j])*(level-amplitude[j])/(amplitude[j+1]-amplitude[j])
    return np.r_[z[:1] if high[0] else [], crosses, z[-1:] if high[-1] else []].reshape(-1, 2).tolist()


def geometry(key):
    p = PARTS[key]
    return [START*1000+p[f"local_{s}_z_mm"] for s in ("start", "center", "end")]


def winding(z_mm, definition, station, axis, order, quadrature=(6, 16)):
    coils = [c for c in definition["coils"] if c["station_index"] == station and c["group"] == axis]
    nodes, dl, _ = _unit_nodes(json.dumps(coils, sort_keys=True), definition["turn_count_per_coil"], *quadrature)
    phi = np.zeros(1) if order == 1 else np.arange(16)*(2*np.pi/16)
    radius = 0. if order == 1 else .0012
    p = np.empty((len(z_mm), len(phi), 3))
    p[:, :, 0], p[:, :, 1] = radius*np.cos(phi), radius*np.sin(phi)
    p[:, :, 2] = (z_mm*.001-START)[:, None]
    b = _induced_cuda(p.reshape(-1, 3), nodes, dl, winding=True).reshape(p.shape)
    transverse = b[:, :, 0]+1j*b[:, :, 1]
    return np.abs(np.mean(transverse*np.exp(1j*(order-1)*phi), axis=1))/(radius**(order-1))


def main():
    ref = json.loads((EVIDENCE/"normal-objective-probe.json").read_text())
    after = ref["after"]
    drives = np.array([s[1] for s in ref["fixed_windings"]])
    np.testing.assert_array_equal(OWNER[ACTIVE], [s[0] for s in ref["fixed_windings"]])
    field_path = WORK/"normal-core-field.npz"
    if sha256(field_path.read_bytes()).hexdigest() != after["corrector_table_sha256"]:
        raise ValueError("Incorrect corrector field")
    model = Native(field_path, ref["max_step_mm"], ref["source_packets"], after["objective_At"], column_planes())
    # Restrict every comparison to the incident path actually present in the ray diagram.
    z = np.r_[np.arange(START*1000, SURFACE*1000, .5), SURFACE*1000]
    mz = model.m.nodes*1000
    data, profiles, stations = {}, {}, {}

    def add(name, key, center, method, units, value, active, **checks):
        a = np.abs(value)
        profiles[name] = a
        data[name] = dict(key=key, center_mm=center, mechanical_span_mm=geometry(key)[::2], method=method,
                          units=units, active=active, peak=float(a.max()), peak_z_mm=float(z[a.argmax()]),
                          bands_mm={str(t):bands(z, a, t) for t in (.01, .1, .5)}, **checks)
        stations[name] = center

    def unit_profile(definition, station, axis, order, center, grid):
        profile = winding(grid, definition, station, axis, order)
        samples = np.unique(np.clip(np.searchsorted(grid, center + np.arange(-50, 51, 2)),
                                    0, len(grid)-1))
        finer = winding(grid[samples], definition, station, axis, order, quadrature=(12, 24))
        error = float(np.max(abs(profile[samples]-finer))/max(profile))
        if error > .01:
            # Closely spaced poles can require more integration nodes across the winding.
            profile = winding(grid, definition, station, axis, order, quadrature=(12, 24))
            finer = winding(grid[samples], definition, station, axis, order, quadrature=(14, 30))
            error = float(np.max(abs(profile[samples]-finer))/max(profile))
            if error > .01:
                raise ValueError(f"Winding profile changed by {error:.3%} of peak on refinement")
        return profile, error

    for name, key in ROUND.items():
        select = OWNER[ACTIVE] == key
        b = drives[select] @ model.coils[select, :, 0]
        add(name, key, geometry(key)[1], "installed vacuum winding Bz", "T",
            np.interp(z, mz, b), True, winding_A=drives[select].tolist())
    b_ol = model.bg[:, 0] - model.m.group[0, :, 0]
    add("Objective", "objective_lens", geometry("objective_lens")[1], "installed axisymmetric material Bz",
        "T", np.interp(z, mz, b_ol), True)

    for name, key in MATERIAL.items():
        with np.load(CACHE/(key+".npz")) as f:
            if (f["source_sha256"].item() != ref["field_archive_sha256"]["normal-final-corrector-solution.npz"]
                    or f["bias_A"] != after["corrector_A"][key]):
                raise ValueError("Incorrect material response bias")
            plus, minus = f["plus"], f["minus"]
            avg = .5*(plus+minus)
            parent = key.rsplit("_", 1)[0] if key.startswith("probe_hpol") else key
            add(name, parent, geometry(parent)[1], "centered material winding response at final bias",
                "T/A" if f["kind"].item() == "round" else "T/m²/A", np.interp(z, f["z_mm"], abs(avg)), True,
                bias_A=float(f["bias_A"]), perturbation_A=float(f["increment_A"]),
                half_step_asymmetry_relative_to_peak=float(np.max(abs(plus-minus))/(2*max(abs(avg)))),
                radius_check_relative_to_peak=float(np.max(abs(f["inner"]-avg))/max(abs(avg))),
                residuals=f["residuals"].tolist())

    for name, (key, station) in STEERING.items():
        definition = transverse_winding_definition(PARTS[key], PARTS)
        center = next(c["center_z_mm"] for c in definition["coils"] if c["station_index"] == station)+START*1000
        for axis in ("x", "y"):
            profile, error = unit_profile(definition, station, axis, 1, center, z)
            add(name+" "+axis.upper(), key, center, "unit-current vacuum response of unpowered saddle winding",
                "T/A", profile, False, station_index=station, quadrature_check_relative_to_peak=error)
    for name, key in TRIMS.items():
        if key == "condenser_stigmator":
            definition, order = transverse_winding_definition(PARTS[key], PARTS), 2
        else:
            definition = multipole_winding_definition(PARTS[key], PARTS)
            order = definition["order"]
        for axis in ("x", "y"):
            profile, error = unit_profile(definition, 0, axis, order, geometry(key)[1], z)
            add(name+" "+axis.upper(), key, geometry(key)[1], "unit-current vacuum response of undriven trim",
                {2:"T/m/A", 3:"T/m²/A"}[order], profile, False, quadrature_check_relative_to_peak=error)
    for name, key in (("C2 aperture", "condenser_aperture_2"), ("C3 aperture", "condenser_aperture_3")):
        stations[name] = geometry(key)[1]
    stations["Specimen entrance"] = SURFACE*1000
    stations["Virtual scan pivot"] = after["downstream_calibration"]["closest_common_pivot_mm"]
    stations["Two-lens front focus"] = after["downstream_calibration"]["isolated_front_focal_plane_mm"]
    others = {}
    for label, center in stations.items():
        if center < z[0] or center > z[-1]+.001:
            raise ValueError("Station outside registered domain")
        others[label] = dict(z_mm=center, active_profile_fraction_at_center={
            k:float(np.interp(center, z, v)/max(v)) for k, v in profiles.items()
            if data[k]["active"] and (label not in data or data[k]["key"] != data[label]["key"])})
    ranges = {}
    for name, first, last in (("HP2 to HP1", "probe_hp2_hexapole", "probe_hp1_hexapole"),
                              ("ADL to TL12", "adapter_lens", "probe_tl12_lens"),
                              ("Beam deflector to TL12", "beam_deflector", "probe_tl12_lens"),
                              ("ADL to AC exit", "adapter_lens", "ac_deflector")):
        lo, hi = geometry(first)[0], geometry(last)[-1]
        ranges[name] = dict(start_mm=lo, end_mm=hi, length_mm=hi-lo)
    # Follow the same cubic radial fit as Mesh.base over the larger objective field archive.
    # These downstream stations have no full-column trajectory or other incident field profiles.
    objective_path = EVIDENCE/"normal-final-objective-field.npz"
    if sha256(objective_path.read_bytes()).hexdigest() != ref["field_archive_sha256"][objective_path.name]:
        raise ValueError("Incorrect extended objective solution")
    with np.load(objective_path) as ol:
        ol_domain = ol["z"][[0,-1]]*1000
        r = ol["r"]
        sel = (r > 0) & (r <= .00288*(1+1e-14))
        fit = np.polynomial.polynomial.polyvander((r[sel]/.00288)**2, 3)
        axis = np.linalg.lstsq(fit, ol["a_phi"][sel]/r[sel, None], rcond=None)[0][0]
        spline = make_interp_spline(ol["z"], 2*np.sign(after["objective_At"])*axis, k=3)
    np.testing.assert_allclose(spline(model.m.nodes), b_ol, rtol=1e-8, atol=1e-10)
    upper, lower = START*1000+np.asarray(PARTS["image_diffraction_deflector"]["interaction_centers_local_z_mm"])
    exit_z = START*1000+max(part["local_end_z_mm"] for part in PARTS.values())
    z_post = np.r_[np.arange(SURFACE*1000, exit_z, .5), exit_z]
    if z[0] < ol_domain[0]-.001 or exit_z > ol_domain[1]+.001:
        raise ValueError("Objective range extends beyond the archived field")
    z_full = np.r_[z[:-1], z_post]
    full_ol = abs(spline(z_full*.001))
    full_ol_bands = {str(t):bands(z_full, full_ol, t) for t in (.01,.1,.5)}
    downstream = {name: {"center_mm":z0, "objective_Bz_T":float(spline(z0*.001))} for name, z0 in (
        ("Specimen entrance", SURFACE*1000), ("Physical objective aperture", geometry("objective_aperture")[1]),
        ("Image / diffraction upper", upper), ("Image / diffraction lower", lower),
        ("Objective stigmator", geometry("objective_stigmator")[1]))}
    outgoing = {}
    for name, key, station, order, center in (
            ("Image / diffraction upper","image_diffraction_deflector",0,1,upper),
            ("Image / diffraction lower","image_diffraction_deflector",1,1,lower),
            ("Objective stigmator","objective_stigmator",0,2,geometry("objective_stigmator")[1])):
        definition = transverse_winding_definition(PARTS[key], PARTS)
        for axis in ("x","y"):
            v, err = unit_profile(definition, station, axis, order, center, z_post)
            outgoing[name+" "+axis.upper()] = dict(center_mm=center, mechanical_span_mm=geometry(key)[::2],
                units="T/A" if order==1 else "T/m/A", peak=float(max(v)),
                quadrature_check_relative_to_peak=err, bands_mm={str(t):bands(z_post,v,t) for t in (.01,.1,.5)})
    with np.load(model.m.cache/"electric.npz") as elec:
        ze, v = elec["z"], elec["voltage"][0]
        electric_mask = (ze[:-1]<ranges["ADL to AC exit"]["end_mm"]*.001) & (ze[1:]>ranges["ADL to AC exit"]["start_mm"]*.001)
        axis_e = float(np.max(np.abs(np.diff(v)/np.diff(ze))[electric_mask]))
    npz = EVIDENCE/"field-footprint-profiles.npz"
    _save(npz, z_mm=z, names=np.array(list(profiles)), amplitude=np.stack(list(profiles.values())))
    write(EVIDENCE/"field-footprints.json", dict(
        reference="normal-objective-probe.json / after; registered incident path from gun exit to specimen entrance",
        z_domain_mm=z[[0,-1]].tolist(), grid_mm=.5, field_probes_radius_mm=[.6, 1.2],
        source_table_sha256=after["corrector_table_sha256"], profile_file_sha256=sha256(npz.read_bytes()).hexdigest(),
        mechanical_extents=ranges, components=data, stations=others,
        objective_field_at_downstream_stations=downstream, objective_incident_and_outgoing_bands_mm=full_ol_bands,
        outgoing_z_domain_mm=z_post[[0,-1]].tolist(), outgoing_unit_windings=outgoing,
        max_abs_axis_E_in_ADL_to_AC_V_per_m=axis_e))


if __name__ == "__main__":
    main()

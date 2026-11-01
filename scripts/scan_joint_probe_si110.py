"""Stage raster of the archived joint probe through a classical Si[110] elastic screen.

This offline observable uses ideal specimen-exit angular detectors. The material
model and its limitations are documented with the images; no wave is constructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tomllib

import numpy as np
from ase.io import read
from numba import njit, prange
from scipy.special import roots_legendre
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from temsim.specimen.elastic_transport import (
    screened_rutherford_angle_cdf, screened_rutherford_parameter,
    screened_rutherford_total_cross_section_cm2,
)
from temsim.specimen.geometry import quaternion_from_zone_axes, quaternion_to_matrix

PROBE = ROOT / "docs/development/probe_joint_20261101/extend-joint-01-focus1"
CIF = ROOT / "tests/fixtures/cif/Si.cif"
PRESET = ROOT / "configs/specimens/10_si_110.toml"
OUT = ROOT / "docs/development/si110_joint_probe_scan_20261101"
SURFACE_HASH = "eec94f0387c858beb6d58c76a9c5997ec9bb47477809b32f92bb837fb256f732"
INPUT_ID = "28a3643f3ea3e27538d78ee8fdc1cebdf19b42802bd0c3733be314b2bf3af53e"
RANGES = ((0., 15.), (50., 150.), (15., 50.), (150., 1000.*np.pi))
PIXELS, PITCH, THICKNESS = 64, .020, 5.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_probe():
    report = json.loads((PROBE / "result.json").read_text())
    if sha(PROBE / "surface.npz") != SURFACE_HASH or report["input_digest"] != INPUT_ID:
        raise ValueError("The selected reference changed")
    with np.load(PROBE / "surface.npz", allow_pickle=False) as p:
        phase = {k: p[k] for k in p.files}
    with np.load(PROBE / "current.npz", allow_pickle=False) as p:
        currents = p["particle_current_a"]
        np.testing.assert_array_equal(p["surviving_source_ids"], phase["source_id"])
        np.testing.assert_array_equal(p["xy_nm"], np.stack((phase["x_m"], phase["y_m"]), axis=1)*1e9)
        np.testing.assert_allclose(currents, phase["weights"] * float(p["emitted_current_a"]), rtol=2e-15)
        np.testing.assert_allclose(p["z_mm"], report["at_specimen"]["z_mm"], rtol=0, atol=1e-12)
    with np.load(PROBE.parents[1] / "probe_stage12_20261101/inputs/source-2000.npz") as source:
        for key in ("source_id", "weights"):
            np.testing.assert_array_equal(phase[key], source[key])
    assert len(np.unique(phase["source_id"])) == len(currents) == 2000
    assert np.all(phase["alive"] & (currents > 0))
    return phase, currents, report


def lattice():
    """Four-site orthorhombic [110] repeat, reduced from the actual cubic CIF."""
    atoms = read(CIF)
    np.testing.assert_allclose(atoms.cell.angles(), 90.)
    np.testing.assert_allclose(atoms.cell.lengths(), atoms.cell.lengths()[0])
    assert len(atoms) == 8 and np.all(atoms.numbers == 14)
    a = atoms.cell.lengths()[0] * .1
    repeat = np.array([a/np.sqrt(2), a, a/np.sqrt(2)])
    rot = quaternion_to_matrix(quaternion_from_zone_axes(atoms.cell.array, (1,1,0), (1,-1,0)))
    f = np.round((atoms.positions @ rot.T) * .1 / repeat, 12) % 1.
    basis = np.unique(f, axis=0) * repeat
    assert len(basis) == 4
    np.testing.assert_allclose(len(basis) / repeat.prod(), len(atoms) / (a**3))
    # Centre the foil at the crystal origin. z coordinates below are from its entrance.
    planes = []
    for dx, dy, dz in basis:
        for n in range(-math.ceil(THICKNESS/repeat[2])-1, math.ceil(THICKNESS/repeat[2])+2):
            z = .5*THICKNESS + dz + n*repeat[2]
            if 0 <= z < THICKNESS:
                planes.append((dx, dy, z))
    return repeat, basis, np.array(sorted(planes, key=lambda r: r[2]))


@njit(parallel=True, cache=True)
def ray_depths(scan_xy, xy, slopes, area, sites, repeat, sigma, thickness):
    """Line integral of normalized 3-D Gaussian atomic number density per ray."""
    depths = np.zeros((len(scan_xy), len(area)))
    for i in prange(len(scan_xy)):
        for j in range(len(area)):
            sx, sy = slopes[j]
            dz_dir = 1. / math.sqrt(1. + sx*sx + sy*sy)
            dx_dir, dy_dir = sx*dz_dir, sy*dz_dir
            distance = thickness/dz_dir
            x0, y0 = scan_xy[i, 0]+xy[j, 0], scan_xy[i, 1]+xy[j, 1]
            density = 0.
            for k in range(len(sites)):
                ax, ay, z = sites[k]
                dx, dy = ax-(x0+sx*z), ay-(y0+sy*z)
                dx -= round(dx/repeat[0])*repeat[0]
                dy -= round(dy/repeat[1])*repeat[1]
                along = dx*dx_dir + dy*dy_dir
                b2 = dx*dx + dy*dy - along*along
                if b2 < (8.*sigma)**2:
                    centre = z/dz_dir + along
                    inv = 1./(math.sqrt(2.)*sigma)
                    mass = .5*(math.erf((distance-centre)*inv)+math.erf(centre*inv))
                    density += math.exp(-b2/(2.*sigma*sigma))*mass
            depths[i,j] = density*area[j]/(2.*math.pi*sigma*sigma)
    return depths


def cap_single(theta, mu, delta):
    """Exact azimuth-integrated screened Rutherford probability inside a lab cone."""
    c = np.cos(theta)
    t = (1.+2.*delta)*c - mu
    return .5*(1.-t/np.hypot(t, np.sqrt(4.*delta*(1.+delta))*np.sin(theta)))


def independent_depth_check(scan_xy, xy, slopes, area, sites, repeat, sigma):
    """Independent Cartesian ray/atom projections with a 3x3 lateral image sum."""
    from scipy.special import ndtr
    start = np.column_stack((xy+scan_xy, np.zeros(len(xy))))
    direction = np.column_stack((slopes,np.ones(len(slopes))))
    direction /= np.linalg.norm(direction,axis=1)[:,None]
    length = THICKNESS/direction[:,2]
    total = np.zeros_like(area)
    for ax,ay,z in sites:
        # Choose the cell around the entry, not the depth-specific wrapped point.
        centre = np.rint((start[:,:2]-[ax,ay])/repeat[:2])*repeat[:2]
        for i in (-1,0,1):
            for j in (-1,0,1):
                site = np.column_stack((ax+centre[:,0]+i*repeat[0],
                                        ay+centre[:,1]+j*repeat[1],np.full(len(xy),z)))
                vec = site-start
                along = (vec*direction).sum(axis=1)
                perp2 = ((vec-along[:,None]*direction)**2).sum(axis=1)
                total += (ndtr((length-along)/sigma)-ndtr(-along/sigma))*np.exp(-.5*perp2/sigma**2)
    return area*total/(2*np.pi*sigma**2)


def event_responses(mu, delta, max_events, order, nquad):
    """Conditional angular window probabilities after n independent elastic rotations.

    Zonal spherical convolution raises each Legendre moment to its nth power.
    Direct rays and the first collision use exact window integrals.
    """
    from numpy.polynomial.legendre import legvander
    x, w = roots_legendre(nquad)
    # Same cross section as the project, evaluated at each individual energy.
    density = .5*delta[:,None]*(1.+delta[:,None])/((1.-x[None,:])*.5+delta[:,None])**2
    moments = (density*w) @ legvander(x, order)
    np.testing.assert_allclose(moments[:,0], 1., rtol=0, atol=2e-8)
    moments[:,0] = 1.  # Unit zeroth moment is exact.
    p_mu = legvander(mu, order)
    theta = np.arccos(mu)
    table = np.empty((len(RANGES), len(mu), max_events+1))
    for k,(lo,hi) in enumerate(RANGES):
        low, high = lo*.001, hi*.001
        table[k,:,0] = (theta >= low) & (theta < high)
        table[k,:,1] = cap_single(high, mu, delta)-cap_single(low, mu, delta)
        p_lo, p_hi = legvander(np.cos([low,high]), order+1)
        area = np.r_[.5*(np.cos(low)-np.cos(high)),
                     .5*((p_hi[:-2]-p_hi[2:])-(p_lo[:-2]-p_lo[2:]))]
        coeff = p_mu*area
        power = moments.copy()
        for n in range(2, max_events+1):
            power *= moments
            table[k,:,n] = np.einsum("ij,ij->i", coeff, power)
    np.testing.assert_allclose(table.sum(axis=0), 1., rtol=0, atol=1e-11)
    if table.min() < -1e-10 or table.max() > 1.+1e-10:
        raise ValueError("Invalid angular probability; increase angular integration order")
    return np.ascontiguousarray(table)


@njit(parallel=True, cache=True)
def collect(depths, current, table):
    frames = np.zeros((len(depths), table.shape[0]))
    multiple = np.zeros(len(depths))
    for i in prange(len(depths)):
        for j in range(len(current)):
            tau = depths[i,j]
            p = math.exp(-tau)
            multiple[i] += current[j]*(1.-p*(1.+tau))
            for n in range(table.shape[2]):
                for k in range(table.shape[0]):
                    frames[i,k] += current[j]*p*table[k,j,n]
                p *= tau/(n+1)
                if p < 1e-18 and n+1 >= tau:
                    break
    return frames, multiple


def render(folder, coords, frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tifffile
    edges = [coords[0]-.5*PITCH, coords[-1]+.5*PITCH]*2
    names = ["BF", "Annular DF"]
    filenames = ["bf", "df"]
    def panel(fig, ax, k):
        # Keep raw linear minimum/maximum and exact pixels; no filtering or interpolation.
        scale,unit = ((1e6,"µA"),(1e9,"nA"))[k]
        im = ax.imshow(frames[k]*scale, origin="lower", extent=edges, interpolation="nearest",
                       cmap="gray", aspect="equal")
        ax.set_title(f"{names[k]}  |  {RANGES[k][0]:.0f}–{RANGES[k][1]:.0f} mrad")
        ax.set_xlabel("x  [1 −1 0]  (nm)")
        ax.set_ylabel("y  [0 0 −1]  (nm)")
        bar = fig.colorbar(im, ax=ax, fraction=.046, pad=.035)
        bar.set_label(f"Collected current ({unit})")
    for k,name in enumerate(filenames):
        fig, ax = plt.subplots(figsize=(5,4.15), layout="constrained")
        panel(fig,ax,k)
        fig.savefig(folder/f"{name}.png", dpi=180)
        plt.close(fig)
        tifffile.imwrite(folder/f"{name}.tiff", frames[k], photometric="minisblack",
                         metadata={"axes":"YX", "unit":"A", "row_zero":"minimum sample y",
                                   "pixel_size_nm":PITCH, "first_pixel_center_nm":float(coords[0]),
                                   "detector_mrad":RANGES[k], "scan":"stage-relative"})
    fig, axes = plt.subplots(1,2,figsize=(9.9,4.6),layout="constrained")
    for k,ax in enumerate(axes):
        panel(fig,ax,k)
    fig.suptitle("Si [110]   |   5 nm   ·   300 kV   ·   29.72 mrad incident probe\n"
                 "Classical elastic scan   ·   virtual exit-angle detectors", fontsize=13)
    fig.savefig(folder/"bf-df.png",dpi=180)
    plt.close(fig)


def central_line(folder, coords, maps, basis, repeat):
    """Read one native column nearest x=0 on its positive side; do not fit peaks."""
    ix = int(np.searchsorted(coords,0.))
    same_x = basis[np.isclose(basis[:,0],0),1]
    projected_pair = np.sort((same_x+.5*repeat[1])%repeat[1]-.5*repeat[1])
    separation = np.diff(projected_pair).item()
    df = maps[1,:,ix]
    peaks = []
    for y in projected_pair:
        indices = np.flatnonzero(np.abs(coords-y) < .25*separation)
        k = indices[np.argmax(df[indices])]
        assert k > indices[0] and k < indices[-1] and df[k] > max(df[k-1],df[k+1])
        peaks.append(k)
    valley = peaks[0] + np.argmin(df[peaks[0]:peaks[1]+1])
    assert peaks[0] < valley < peaks[1]
    np.savetxt(folder/"native-profile.csv", np.column_stack((coords,maps[0,:,ix],df)),
               delimiter=",",header="y_nm,bf_current_A,df_current_A",comments="")
    return dict(x_nm=float(coords[ix]), cif_y_nm=projected_pair.tolist(),
                cif_pair_nm=separation, df_native_peak_y_nm=coords[peaks].tolist(),
                df_native_spacing_nm=float(coords[peaks[1]]-coords[peaks[0]]),
                df_peak_na=(df[peaks]*1e9).tolist(), df_saddle_y_nm=float(coords[valley]),
                df_saddle_na=float(df[valley]*1e9))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    folder = parser.parse_args().output
    folder.mkdir(exist_ok=True, parents=True)
    p, current, parent = load_probe()
    repeat, basis, sites = lattice()
    sigma = tomllib.loads(PRESET.read_text())["atomistic"]["thermal_sigma_angstrom"]*.1
    coords = (np.arange(PIXELS)-(PIXELS-1)/2)*PITCH
    positions = np.stack(np.meshgrid(coords, coords), axis=-1).reshape(-1,2)
    xy = np.column_stack((p["x_m"],p["y_m"]))*1e9
    slopes = np.column_stack((p["tx_rad"],p["ty_rad"]))
    mu = 1./np.sqrt(1.+(slopes**2).sum(axis=1))
    assert 8*sigma/min(mu) < .5*min(repeat[:2])   # Only one periodic image per site can contribute.
    delta = np.array([screened_rutherford_parameter(14,e) for e in p["kinetic_energy_ev"]])
    area = np.array([screened_rutherford_total_cross_section_cm2(14,e)*1e14 for e in p["kinetic_energy_ev"]])
    print("Integrating original ray paths through the oriented CIF", flush=True)
    depths = ray_depths(positions, xy, slopes, area, sites, repeat, sigma, THICKNESS)
    n = max(2, int(poisson.isf(1e-14,depths.max())))
    probes = np.array([0, 1159, 2096, 3421, 4095])
    shifted = ray_depths(positions[probes]+repeat[:2], xy, slopes, area, sites, repeat, sigma, THICKNESS)
    periodic_error = float(abs(shifted-depths[probes]).max())
    assert periodic_error < 1e-10
    independent = independent_depth_check(positions[2096],xy,slopes,area,sites,repeat,sigma)
    depth_error = float(abs(independent-depths[2096]).max())
    assert depth_error < 1e-10
    print(f"Maximum ray optical depth {depths.max():.5g}; angular orders 0 to {n}",flush=True)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4, user_api="blas"):
        fine = event_responses(mu,delta,n,768,1536)
        coarse = event_responses(mu,delta,n,512,1024)
    angular_change = float(abs(fine-coarse).max())
    assert angular_change < 1e-5
    # Independent axial CDF check for the analytic off-axis detector integral.
    for k in (0,1):
        lo, hi = np.asarray(RANGES[k])*.001
        e = p["kinetic_energy_ev"][0]
        ref = screened_rutherford_angle_cdf(hi,14,e)-screened_rutherford_angle_cdf(lo,14,e)
        assert abs(ref-(cap_single(hi,1.,delta[0])-cap_single(lo,1.,delta[0]))) < 1e-10
    values, plural = collect(depths,current,fine)
    old, _ = collect(depths,current,coarse)
    frame_change = float(abs(values-old).max())
    assert frame_change < 1e-12
    vacuum, _ = collect(np.zeros((1,len(current))), current, fine)
    direct = np.array([current[(np.arctan(np.linalg.norm(slopes,axis=1))*1000 >= lo)
                              & (np.arctan(np.linalg.norm(slopes,axis=1))*1000 < hi)].sum()
                       for lo, hi in RANGES])
    np.testing.assert_allclose(vacuum[0],direct,rtol=0,atol=2e-18)
    assert direct[1] == 0. and np.min(values) >= 0 and np.all(np.isfinite(values))
    ledger_error = float(abs(values.sum(axis=1)-current.sum()).max())
    assert ledger_error < 2e-17
    maps = values.T.reshape(len(RANGES),PIXELS,PIXELS)
    np.savez_compressed(folder/"scan.npz", x_nm=coords, y_nm=coords, bf_a=maps[0], df_a=maps[1],
        gap_15_to_50_a=maps[2], other_150mrad_to_pi_a=maps[3], plural_a=plural.reshape(PIXELS,PIXELS),
        vacuum_window_a=vacuum[0], windows_mrad=np.array(RANGES), incident_current_a=current.sum())
    render(folder,coords,maps)
    line = central_line(folder,coords,maps,basis,repeat)
    import tifffile
    for k,name in enumerate(("bf","df")):
        np.testing.assert_array_equal(tifffile.imread(folder/f"{name}.tiff"), maps[k])
    metrics = dict(probe=PROBE.relative_to(ROOT).as_posix(), input_digest=INPUT_ID, source_digest=parent["source_digest"],
        surface_sha256=SURFACE_HASH, current_sha256=sha(PROBE/"current.npz"), cif=CIF.relative_to(ROOT).as_posix(),
        cif_sha256=sha(CIF), specimen_preset_sha256=sha(PRESET), original=len(current), surviving=int(p["alive"].sum()),
        effective=float(current.sum()**2/(current@current)), incident_a=float(current.sum()), entrance_z_mm=parent["at_specimen"]["z_mm"],
        incident_alpha95_mrad=parent["at_specimen"]["alpha95_mrad"], max_incident_axis_angle_mrad=float(np.arccos(mu.min())*1000),
        incident_d95_nm=parent["at_specimen"]["d95_nm"], energy_range_ev=[p["kinetic_energy_ev"].min(),p["kinetic_energy_ev"].max()],
        scan_pixels_yx=[PIXELS,PIXELS], pixel_nm=PITCH, field_width_nm=PIXELS*PITCH, axes_uvw=[[1,-1,0],[0,0,-1],[1,1,0]],
        lateral_boundary="infinite periodic crystal; all original probe tails included", thickness_nm=THICKNESS,
        periods_xyz_nm=repeat.tolist(), oriented_cell_sites_xyz_nm=basis.tolist(), slab_sites_per_lateral_repeat=len(sites),
        projected_sites_xy_nm=np.unique(sites[:,:2],axis=0).tolist(), atomic_thermal_sigma_nm=sigma,
        detector="ideal exit-angle masks about fixed laboratory +z; stage raster", model="classical straight-path atomic-density elastic screen with compound Poisson angular redistribution",
        coherent=False, shot_noise=False, max_ray_optical_depth=float(depths.max()), max_poisson_order=n,
        max_omitted_poisson_mass=float(poisson.sf(n,depths.max())), legendre_degrees=[512,768], quadrature_orders=[1024,1536],
        max_conditional_angular_change=angular_change, max_detector_refinement_change_a=frame_change, max_pixel_current_balance_error_a=ledger_error,
        periodic_translation_max_tau_error=periodic_error, independent_cartesian_max_tau_error=depth_error,
        native_line=line, max_plural_incident_fraction=float(plural.max()/current.sum()),
        vacuum_window_a=vacuum[0].tolist(), windows_mrad=RANGES, specimen_current_a_by_window={
            name:dict(min=float(m.min()), mean=float(m.mean()), max=float(m.max())) for name,m in zip(("bf","df","gap","beyond_df"),maps)})
    (folder/"metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps(metrics["specimen_current_a_by_window"]),flush=True)


if __name__ == "__main__":
    main()

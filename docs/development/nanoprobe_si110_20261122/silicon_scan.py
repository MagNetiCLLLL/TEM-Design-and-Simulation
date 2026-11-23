"""Run a stage-relative Si [110] diagnostic with the unmodified entrance particles."""
import argparse
import json
from pathlib import Path
import sys
import tomllib

import numpy as np
from scipy.stats import poisson
from threadpoolctl import threadpool_limits
from numba import get_num_threads, set_num_threads

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]/"scripts"))
import scan_joint_probe_si110 as scan


def run(folder):
    report = json.loads((folder/"tracking.json").read_text())
    with np.load(folder/"column-and-landings.npz") as col, np.load(report["gun_source"]) as gun:
        ids = col["source_ids"]
        np.testing.assert_array_equal(ids, gun["ids"])
        np.testing.assert_array_equal(col["weights"], gun["weight"])
        assert len(np.unique(ids)) == len(ids) and (col["weights"]>0).all()
        mask = col["loss"] == -1
        assert (col["loss"]>=-1).all() and mask.sum()>=2000
        phase = col["surface_phase"][mask]
        joule_ev = 300000.+gun["energy_offset_ev"][mask]
        current = col["weights"][mask]*float(col["source_A"])
    np.testing.assert_allclose(current.sum(), report["surface"]["specimen_uA"]*1e-6, rtol=2e-13)
    output = folder/"si110"
    output.mkdir(exist_ok=True)
    repeat, basis, sites = scan.lattice()
    sigma = tomllib.loads(scan.PRESET.read_text())["atomistic"]["thermal_sigma_angstrom"]*.1
    xy, slope = phase[:,:2]*1e9, phase[:,2:4]
    mu = 1/np.sqrt(1+np.sum(slope*slope,axis=1))
    area = np.array([scan.screened_rutherford_total_cross_section_cm2(14,x)*1e14 for x in joule_ev])
    delta = np.array([scan.screened_rutherford_parameter(14,x) for x in joule_ev])
    coords = (np.arange(scan.PIXELS)-.5*(scan.PIXELS-1))*scan.PITCH
    positions = np.stack(np.meshgrid(coords,coords),axis=-1).reshape(-1,2)
    assert 8*sigma/min(mu) < min(repeat[:2])/2
    # Bound ray optical depth by one projected column in each z-layer.
    norder = max(2,int(poisson.isf(1e-15,len(sites)*max(area)/(2*np.pi*sigma**2))))
    set_num_threads(min(12,get_num_threads()))
    all_values = np.zeros((len(positions),len(scan.RANGES)))
    halves = np.zeros((2,len(positions),len(scan.RANGES)))
    midpoint = len(current)//2
    largest_tau, depth_error, angular_error = 0., 0., 0.
    for half,(lo,hi) in enumerate(((0,midpoint),(midpoint,len(current)))):
        for i in range(lo,hi,512):
            sl = slice(i,min(i+512,hi))
            with threadpool_limits(4,"blas"):
                fine = scan.event_responses(mu[sl],delta[sl],norder,768,1536)
                if i==0:
                    coarse = scan.event_responses(mu[sl],delta[sl],norder,512,1024)
                    angular_error = float(abs(fine-coarse).max())
                    assert angular_error < 1e-5
            depth = scan.ray_depths(positions,xy[sl],slope[sl],area[sl],
                                    sites,repeat,sigma,scan.THICKNESS)
            if i==0:
                cart = scan.independent_depth_check(positions[2096],xy[sl],slope[sl],area[sl],sites,repeat,sigma)
                depth_error = float(np.max(np.abs(cart-depth[2096])))
                assert depth_error < 1e-9
            values,_ = scan.collect(depth,current[sl],fine)
            halves[half] += values
            largest_tau = max(largest_tau,float(depth.max()))
        print("Completed finite-source scan subset",half+1,flush=True)
    all_values = halves.sum(axis=0)
    balance = float(abs(all_values.sum(axis=1)-current.sum()).max())
    assert balance < 2e-17
    maps=all_values.T.reshape(len(scan.RANGES),scan.PIXELS,scan.PIXELS)
    np.savez_compressed(output/"detectors.npz",coordinates_nm=coords,all_window_A=maps,
                        half_window_A=halves,half_incident_A=[current[:midpoint].sum(),current[midpoint:].sum()])
    scan.render(output,coords,maps,report["surface"]["alpha95_mrad"])
    line=scan.central_line(output,coords,maps,basis,repeat)
    # Each normalized subset is used only to compare independently sampled contrast.
    df=halves[:,:,1]/np.array([current[:midpoint].sum(),current[midpoint:].sum()])[:,None]
    centered=df-df.mean(axis=1)[:,None]
    norm=np.linalg.norm(centered,axis=1)
    metrics=dict(source_particles=len(ids),live=len(current),effective=float(current.sum()**2/(current@current)),
        current_A=float(current.sum()),surface=report["surface"],entrance_mm=report["surface_z_mm"],
        entrance_file_sha256=scan.sha(folder/"column-and-landings.npz"),gun_file_sha256=scan.sha(Path(report["gun_source"])),
        cif_sha256=scan.sha(scan.CIF),foil_nm=scan.THICKNESS,pixel_nm=scan.PITCH,pixels=scan.PIXELS,
        detector_windows_mrad=scan.RANGES,model="Classical elastic periodic Si [110], translated specimen, ideal exit-angle detectors",
        independent_depth_error=depth_error,max_depth=largest_tau,max_angular_quadrature_error=angular_error,
        max_omitted_scattering_probability=float(poisson.sf(norder,largest_tau)),pixel_current_balance_A=balance,
        line=line,df_rms_contrast=float(maps[1].std()/maps[1].mean()),
        disjoint_source_df_correlation=float(centered[0]@centered[1]/norm.prod()) if norm.min()>0 else None,
        half_df_relative_rms_difference=float(np.sqrt(np.mean((df[0]-df[1])**2))/df.mean()),
        pair_spacing_nm=line["cif_pair_nm"])
    (output/"scan.json").write_text(json.dumps(metrics,indent=2,allow_nan=False)+"\n")
    print(json.dumps(metrics["line"]),metrics["disjoint_source_df_correlation"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    run(parser.parse_args().folder.resolve())

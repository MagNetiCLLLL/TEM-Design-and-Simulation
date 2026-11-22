"""Locate the brightest patch independently of the current centroid at fixed lens currents."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from fit_threefold import EVIDENCE, HERE, angle_groups, read_sample
from plot_aperture_angle_spots import colour_bar, colour_scale, show_map
from plot_specimen_probe import landing_current_density


OUTPUT = HERE.parents[2] / "outputs/cescor-a2-20261121/spherical"
PITCH_UM = .005
CASE_LABELS = {
    "balanced-native": "Linear interpolation  ·  0.30 mm field grid",
    "balanced-halfstep": "Half integration step",
    "balanced-prefix": "Half source population",
    "balanced-fine-xy": "Finer field sampling",
    "balanced-mesh5": "Coarser material mesh",
    "centre-cubic": "Cubic field interpolation",
    "centre-cubic-fine": "Cubic interpolation  ·  0.15 mm field grid",
    "centre-cubic-mesh5": "Cubic interpolation on the coarser material mesh",
    "centre-hex-off": "HP and HPol currents off",
    "centre-rotated": "Corrector field rotated 180°",
}


def make_rotated_table(source, destination):
    """Rotate the registered vector field by half a turn without resampling it."""
    from temsim.corrector_field_cli import _save

    with np.load(source, allow_pickle=False) as data:
        values = dict(data)
    np.testing.assert_allclose(values["xy_m"], -values["xy_m"][::-1], rtol=0, atol=1e-17)
    source_field = values["B_zyx_T"]
    rotated = source_field[:, ::-1, ::-1].copy()
    rotated[..., :2] *= -1
    np.testing.assert_array_equal(-rotated[:, ::-1, ::-1, :2], source_field[..., :2])
    np.testing.assert_array_equal(rotated[:, ::-1, ::-1, 2], source_field[..., 2])
    values["B_zyx_T"] = rotated
    report = json.loads(str(values["report"]))
    report.update(diagnostic_field_rotation_deg=180., original_table_sha256=sha256(source.read_bytes()).hexdigest())
    values["report"] = json.dumps(report, sort_keys=True)
    _save(destination, **values)


def features(sample, sigma_um=.025):
    xy, current = sample.xy_um, sample.current_na
    edges = np.linspace(-.7,.7,281)
    centres = .5*(edges[:-1]+edges[1:])
    bins = np.histogram2d(*xy.T, bins=(edges,edges),weights=current)[0]
    density = gaussian_filter(bins, sigma_um/PITCH_UM, mode="constant")
    ix, iy = np.unravel_index(np.argmax(density),density.shape)
    peak = centres[[ix,iy]]
    centroid = np.average(xy,axis=0,weights=current)
    result = dict(centroid_nm=(centroid*1000).tolist(), peak_nm=(peak*1000).tolist(),
                  peak_radius_nm=float(1000*np.linalg.norm(peak)), kernel_sigma_nm=sigma_um*1000)
    for label, location in (("peak",peak),("axis",np.zeros(2)),("opposite",-peak)):
        selection = np.linalg.norm(xy-location,axis=1)<=.05
        grouped = [float(current[selection & group].sum()) for group in angle_groups(sample)]
        result[label+"_disk_nA"] = float(current[selection].sum())
        result[label+"_disk_by_angle_nA"] = grouped
        np.testing.assert_allclose(sum(grouped),result[label+"_disk_nA"],rtol=1e-12)
    xy_complex = xy[:,0]+1j*xy[:,1]
    locations = complex(*peak)*np.exp(2j*np.pi*np.arange(3)/3)
    result["patches_120deg"] = [
        dict(x_nm=float(z.real*1000),y_nm=float(z.imag*1000),
             disk_nA=float(current[np.abs(xy_complex-z)<=.05].sum())) for z in locations]
    # Measure departure from 120-degree rotational symmetry, avoiding a single
    # arbitrary peak location in a nearly symmetric ring or three-lobed spot.
    x,y = np.meshgrid(centres,centres,indexing="ij")
    inner = np.hypot(x,y)<.45
    theta = 2*np.pi/3
    xr,yr = np.cos(theta)*x-np.sin(theta)*y, np.sin(theta)*x+np.cos(theta)*y
    at_rotated = map_coordinates(density, [(xr[inner]-centres[0])/PITCH_UM,(yr[inner]-centres[0])/PITCH_UM],
                                 order=1,mode="constant")
    result["rotation120_fractional_difference"] = float(np.sum(abs(density[inner]-at_rotated))/
                                                        np.sum(density[inner]+at_rotated))
    return result


def plot(samples, results):
    panels = ("balanced-native","centre-cubic-fine","centre-rotated","centre-hex-off")
    width = .5
    maps = [landing_current_density(samples[key].xy_um,samples[key].current_na,width,.0125)[0]
            for key in panels]
    colours = colour_scale(maps)
    fig = plt.figure(figsize=(9.7,8.1),facecolor="white")
    fig.text(.082,.959,"What moves the bright patch?",fontsize=19,fontweight="bold",va="top",color="#17212c")
    fig.text(.082,.907,"Specimen upper surface   ·   fixed round-lens currents   ·   100 µm C2 aperture",
             fontsize=10.5,va="top",color="#465362")
    grid=fig.add_gridspec(2,3,width_ratios=(1,1,.041),left=.079,right=.886,
                          top=.815,bottom=.121,hspace=.36,wspace=.26)
    for index,(key,values) in enumerate(zip(panels,maps)):
        ax=fig.add_subplot(grid[index//2,index%2])
        image=show_map(ax,values,width,[-.5,-.25,0,.25,.5],colours)
        peak = np.array(results[key]["peak_nm"])*.001
        ax.plot(0,0,marker="+",ms=12,mew=1.5,color="#42dbff",ls="none")
        ax.plot(*peak,marker="o",ms=9,mfc="none",mec="white",mew=1.2,ls="none")
        if key == "centre-cubic-fine":
            for neighbour in results[key]["patches_120deg"][1:]:
                ax.plot(.001*neighbour["x_nm"],.001*neighbour["y_nm"],marker="o",ms=8,
                        mfc="none",mec="#c6d2dd",mew=.9,ls="none")
        ax.set_title(CASE_LABELS[key],fontsize=11,pad=6)
        if index%2==0:
            ax.set_ylabel("y (µm)")
        if index>=2:
            ax.set_xlabel("x (µm)")
    colour_bar(fig,image,fig.add_subplot(grid[:,2]))
    fig.text(.082,.042,"Cross   optical axis     ·     Circle   peak at 25 nm smoothing     ·     Upper right   all three 120° positions",
             fontsize=10,color="#435466")
    fig.savefig(HERE/"probe-centring.png",dpi=175)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotate-field",nargs=2,type=Path,metavar=("SOURCE","DESTINATION"))
    args = parser.parse_args()
    if args.rotate_field:
        make_rotated_table(*args.rotate_field)
        return
    summary=json.loads((EVIDENCE/"comparison.json").read_text())
    samples,results={},{}
    base=json.loads((OUTPUT/"balanced-native.json").read_text())
    baseline_ids=None
    for key in CASE_LABELS:
        row=json.loads((OUTPUT/(key+".json")).read_text())
        file=OUTPUT/row["phase_file"]
        assert sha256(file.read_bytes()).hexdigest()==row["trajectory_sha256"]
        for param in ("objective_At","currents_A","tl12_gain","mini_gain"):
            np.testing.assert_equal(row[param],base[param])
        with np.load(file) as profile:
            if baseline_ids is None:
                baseline_ids = {k:profile[k] for k in ("ids","weights","energy_offset_eV")}
            n=len(profile["ids"])
            for k in ("ids","energy_offset_eV"):
                np.testing.assert_array_equal(profile[k],baseline_ids[k][:n])
            np.testing.assert_allclose(profile["weights"],baseline_ids["weights"][:n]*len(baseline_ids["ids"])/n)
        if key != "balanced-prefix":
            assert row["source_digest"]==base["source_digest"]
        if key != "centre-hex-off":
            assert row["corrector_field"]["currents_A"]==base["corrector_field"]["currents_A"]
        if key=="centre-rotated":
            assert row["corrector_field"]["original_table_sha256"]==base["corrector_sha256"]
        sample=read_sample(file,row,summary)
        samples[key]=sample
        record=features(sample)
        record.update(surface=row["surface"],through_focus=row["through_focus"],survived=row["survived"],
                      fit=row["fit"],fit_inside_30mrad=row["fit_inside_30mrad"],
                      emitted=row["emitted"],source_sha256=row["source_digest"],trajectory_sha256=row["trajectory_sha256"],
                      field_sha256=row["corrector_sha256"],model_currents_A=row["corrector_field"]["currents_A"],
                      interpolation=row.get("transverse_interpolation","linear"),
                      centre_by_angle_nm=[(1000*np.average(sample.xy_um[g],axis=0,weights=sample.current_na[g])).tolist()
                                          for g in angle_groups(sample)])
        results[key]=record
    sensitivity={str(s):features(samples["balanced-native"],s*.001)["peak_nm"] for s in (15,25,50)}
    final_key = "centre-cubic-fine"
    archived = EVIDENCE/"cs-balanced-cubic.npz"
    copyfile(OUTPUT/(final_key+".npz"), archived)
    results[final_key]["trajectory_file"] = archived.name
    out=dict(histogram_pitch_nm=PITCH_UM*1000,top_hat_disk_radius_nm=50,angle_edges_mrad=[0,10,20,30,None],
             peak_localisation_by_sigma_nm=sensitivity,cases=results)
    (EVIDENCE/"probe-centring.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    plot(samples,results)
    from publish_cs import final_probe_detail

    final_probe_detail(samples[final_key], results[final_key], filename="spherical-probe-fine.png",
                       sampling="cubic, 0.15 mm field grid")


if __name__=="__main__":
    main()

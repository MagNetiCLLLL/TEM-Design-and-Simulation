"""Continue the balanced probe's A2 trim using cubic interpolation and fixed round-lens currents."""

import argparse
from hashlib import sha256
import json
from shutil import copyfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adjust_a2 import FINE_KEYS, OUT, read_json
from adjust_cs import CS_OUT
from fit_threefold import EVIDENCE, HERE, angle_groups, design, fit, physical_coefficients, read_sample, validation_folds
from plot_aperture_angle_spots import colour_bar, colour_scale, show_map
from plot_specimen_probe import landing_current_density
from probe_centring import features


REFERENCE = "centre-cubic-fine"
RESPONSE_NAMES = ("centre-cubic", "a2-cubic-response-a2-set", "a2-cubic-response-cs-hpol-y")
CANDIDATE_NAMES = ("a2-cubic-null", "a2-cubic-both")


def run(name):
    row = read_json(CS_OUT / (name + ".json"))
    if row["transverse_interpolation"] != "cubic":
        raise ValueError(f"{name} used an unqualified transverse interpolation for this continuation")
    return row


def compare_inputs(reference, row, *, material_control=False):
    for key in ("objective_At", "tl12_gain", "mini_gain", "currents_A"):
        np.testing.assert_equal(row[key], reference[key])
    for face in ("entrance", "exit"):
        np.testing.assert_equal(row["C2_foil"][face]["z_m"], reference["C2_foil"][face]["z_m"])
    if row["emitted"] == reference["emitted"] and row["source_digest"] != reference["source_digest"]:
        raise ValueError("The fixed source checkpoint changed")
    original, field = reference["corrector_field"], row["corrector_field"]
    for key in ("base_circuit_sha256", "circuit_sha256", "exterior_padding_mm", "turns"):
        np.testing.assert_equal(field[key], original[key])
    if not material_control:
        np.testing.assert_equal(field["mesh_id"], original["mesh_id"])
    np.testing.assert_equal(row["objective_solution"]["installed_geometry_digest"],
                            reference["objective_solution"]["installed_geometry_digest"])
    for winding, current in original["currents_A"].items():
        if winding not in FINE_KEYS:
            np.testing.assert_equal(field["currents_A"][winding], current)


def xy_current(row):
    currents = row["corrector_field"]["currents_A"]
    return np.array([currents[key] for key in FINE_KEYS])


def current_proposals():
    rows = [run(n) for n in RESPONSE_NAMES]
    for row in rows:
        compare_inputs(rows[0], row)
    start = xy_current(rows[0])
    changes = np.stack([xy_current(row)-start for row in rows[1:]])
    matrices, targets = [], []
    held_out = [run("a2-cubic-response-"+s) for s in ("a2-trim", "a2-first")]
    errors = {}
    for key in ("fit", "fit_inside_30mrad"):
        values = np.array([row[key]["A2"]["cos_sin_mm"] for row in rows])
        response = np.linalg.solve(changes, values[1:]-values[0]).T
        matrices.append(response)
        targets.append(-values[0])
        errors[key] = []
        for other in held_out:
            compare_inputs(rows[0], other)
            predicted = values[0]+response @ (xy_current(other)-start)
            errors[key].append(dict(currents_A=xy_current(other).tolist(),
                                    residual_mm=(other[key]["A2"]["cos_sin_mm"]-predicted).tolist()))
    both = np.linalg.lstsq(np.vstack(matrices),np.concatenate(targets),rcond=None)[0]
    return dict(start_A=start.tolist(), fixed_current_response_runs=list(RESPONSE_NAMES),
                full_response_mm_per_A=matrices[0].tolist(), cone_response_mm_per_A=matrices[1].tolist(),
                condition_number=float(np.linalg.cond(matrices[0])), independent_response_checks=errors,
                full_null_A=(start+np.linalg.solve(matrices[0],targets[0])).tolist(),
                joint_target_A=(start+both).tolist())


def landing_metrics(sample):
    x, y = sample.xy_um.T
    radius = np.hypot(x,y)
    annulus = (radius >= .10) & (radius < .40)
    unit = (x[annulus]+1j*y[annulus])/radius[annulus]
    angular = np.average(unit**3, weights=sample.current_na[annulus])
    result = features(sample)
    result.update(annulus_100_400_nm=dict(current_nA=float(sample.current_na[annulus].sum()),
                                          H3=float(abs(angular))),
                  disk_100nm_nA=float(sample.current_na[radius<=.10].sum()),
                  by_angle=[])
    for group in angle_groups(sample):
        result["by_angle"].append(dict(count=int(np.sum(group)), current_nA=float(sample.current_na[group].sum()),
                                       disk_100nm_nA=float(sample.current_na[group & (radius<=.10)].sum())))
    return result


def observe(name, reference, *, material_control=False):
    row = run(name)
    compare_inputs(reference,row,material_control=material_control)
    path = CS_OUT/row["phase_file"]
    if sha256(path.read_bytes()).hexdigest() != row["trajectory_sha256"]:
        raise ValueError(f"Unmatched particle data for {name}")
    sample = read_sample(path,row,read_json(EVIDENCE/"comparison.json"))
    record = {k:row[k] for k in ("surface","through_focus","fit","fit_inside_30mrad","emitted","survived","effective",
                                 "max_step_mm","source_digest","corrector_sha256","trajectory_sha256",
                                 "waists_mm","common_waists_mm","current_ledger","C2_foil","transverse_interpolation")}
    field = row["corrector_field"]
    record.update(hpol_xy_A=xy_current(row).tolist(),material_size_mm=field["material_size_mm"],
                  material_mesh_id=field["mesh_id"],landing=landing_metrics(sample))
    return sample,record


def fold_coefficients(sample):
    basis,names = design(sample)
    folds = validation_folds(sample.ids)
    estimates = []
    for k in range(5):
        keep = folds!=k
        c,_ = fit(basis[keep],sample.xy_um[keep],sample.current_na[keep])
        estimates.append(physical_coefficients(c,names)["A2"]["cos_sin_mm"])
    return estimates


def comparison_figure(samples, records):
    width, pitch = .50,.0125
    maps = [landing_current_density(s.xy_um,s.current_na,width,pitch)[0] for s in samples]
    colours = colour_scale(maps)
    fig = plt.figure(figsize=(9.7,5.8), facecolor="white")
    fig.text(.077,.947,"A₂ trim at the specimen upper surface",fontsize=19,fontweight="bold",
             color="#17212c",va="top")
    fig.text(.077,.873,"100 µm C2 aperture   ·   cubic interpolation on a 0.15 mm field grid   ·   fixed lens currents outside HPol",
             fontsize=9.9,color="#465362",va="top")
    grid = fig.add_gridspec(1,3,width_ratios=(1,1,.044),left=.077,right=.900,top=.707,bottom=.198,wspace=.26)
    for index,(data,row,label) in enumerate(zip(maps,records,("Before","After"))):
        ax = fig.add_subplot(grid[0,index])
        image = show_map(ax,data,width,[-.50,-.25,0,.25,.50],colours)
        ax.set(xlabel="x (µm)",ylabel="y (µm)")
        ax.set_title(f"{label}   ·   D95 {row['surface']['D95_nm']:.0f} nm",fontsize=11,pad=8)
    colour_bar(fig,image,fig.add_subplot(grid[0,2]))
    fig.text(.077,.057,"Original particle landings at the same plane and magnification. Colour gives absolute current density.",
             fontsize=9.7,color="#465362")
    fig.savefig(HERE/"a2-cubic-probe.png",dpi=185)
    plt.close(fig)


def publish(selected):
    reference = run(REFERENCE)
    before, old = observe(REFERENCE,reference)
    after,new = observe(selected+"-fine",reference)
    for s,r in ((before,old),(after,new)):
        r["five_training_folds_A2_xy_mm"] = fold_coefficients(s)
    src, dst = CS_OUT/(selected+"-fine.npz"), EVIDENCE/"a2-cubic-adjusted.npz"
    copyfile(src,dst)
    old["trajectory_file"] = "cs-balanced-cubic.npz"
    new["trajectory_file"] = dst.name
    for record in (old,new):
        if sha256((EVIDENCE/record["trajectory_file"]).read_bytes()).hexdigest() != record["trajectory_sha256"]:
            raise ValueError("The archived specimen trajectory differs from its measured phase space")
    with np.load(EVIDENCE/old["trajectory_file"]) as baseline, np.load(dst) as adjusted:
        for key in ("ids","weights","energy_offset_eV"):
            np.testing.assert_array_equal(adjusted[key],baseline[key])
        common=(adjusted["loss"]==-1) & (baseline["loss"]==-1)
        same=np.array_equal(adjusted["loss_owner"],baseline["loss_owner"])
    controls = {}
    for control,(left,right) in {
        "coarse_table":("centre-cubic",selected),
        "half_step":("a2-cubic-start-halfstep",selected+"-halfstep"),
        "half_population":("a2-cubic-start-prefix",selected+"-prefix"),
        "coarser_material":("centre-cubic-mesh5",selected+"-mesh5"),
        "finer_material":("a2-cubic-start-mesh32",selected+"-mesh32"),
    }.items():
        original,trial = run(left),run(right)
        pair={}
        for label,name,raw,drive in (("before",left,original,xy_current(reference)),
                                     ("after",right,trial,xy_current(run(selected)))):
            _,record = observe(name,reference,material_control=control in ("coarser_material","finer_material"))
            np.testing.assert_equal(xy_current(raw),drive)
            pair[label]=record
        np.testing.assert_equal(original["corrector_field"]["mesh_id"], trial["corrector_field"]["mesh_id"])
        np.testing.assert_equal(original["source_digest"],trial["source_digest"])
        controls[control]=pair
    trials={}
    for name in CANDIDATE_NAMES:
        _,trials[name] = observe(name,reference)
    grid_records = {}
    for stem in ("cs-trim","cs-trim-fine","cs-trim-mesh5","a2-cubic-start-mesh32",*CANDIDATE_NAMES,
                 selected+"-fine",selected+"-mesh5",selected+"-mesh32"):
        table = OUT/(stem+"-field.npz")
        with np.load(table) as data:
            xy, z = data["xy_m"], data["z_local_m"]
            grid = dict(file=table.name, transverse_pitch_mm=float(1000*(xy[1]-xy[0])),
                        axial_pitch_mm=float(1000*(z[1]-z[0])))
        grid_records[sha256(table.read_bytes()).hexdigest()] = grid
    for record in (old, new, *trials.values(), *(r for pair in controls.values() for r in pair.values())):
        record["field_table"] = grid_records[record["corrector_sha256"]]
    for pair in (dict(before=old,after=new),*controls.values()):
        for pitch in ("transverse_pitch_mm","axial_pitch_mm"):
            np.testing.assert_equal(pair["before"]["field_table"][pitch],pair["after"]["field_table"][pitch])
    document=dict(source_reference="spherical-working-points.json", previous_centring="probe-centring.json",
                  specimen_upper_surface_z_m=read_json(EVIDENCE/"comparison.json")["specimen_upper_surface_z_m"],
                  aperture_diameter_um=100., source_current_uA=15., selected=selected,
                  objective_At=reference["objective_At"], tl12_gain=reference["tl12_gain"], mini_gain=reference["mini_gain"],
                  round_group_currents_A=reference["currents_A"],
                  main_and_trim_windings_A=run(selected)["corrector_field"]["currents_A"],
                  full_and_30mrad_response=current_proposals(), candidates=trials, before=old, after=new,
                  common_surviving_ids=int(common.sum()), identical_loss_owners=bool(same), paired_controls=controls)
    (EVIDENCE/"a2-cubic-working-point.json").write_text(json.dumps(document,indent=2,allow_nan=False)+"\n",
                                                      encoding="utf-8")
    comparison_figure((before,after),(old,new))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish",choices=CANDIDATE_NAMES,help="Export the fully verified working point")
    args=parser.parse_args()
    if args.publish:
        publish(args.publish)
    else:
        print(json.dumps(current_proposals(),indent=2))

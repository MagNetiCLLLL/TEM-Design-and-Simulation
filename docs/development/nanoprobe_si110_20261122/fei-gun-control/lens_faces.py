"""Resolve archived gun-lens currents by metal face and observe the entry footprint."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from numba import get_num_threads, set_num_threads
from scipy.interpolate import CubicSpline

from gun_field import HERE, WORK, key, study_field_reader
from transport import transport


def surfaces(end, lens_hit, ring):
    """Assign a rim exactly on an end plane to that face."""
    z = end[:, 2]
    radius = np.hypot(end[:, 0], end[:, 1])
    tolerance = 1e-11  # metres, much smaller than the clearance or mesh pitch
    front = lens_hit & (abs(z - ring["start_m"]) < tolerance)
    back = lens_hit & (abs(z - ring["stop_m"]) < tolerance)
    wall = lens_hit & ~front & ~back & (abs(radius - ring["inner_m"]) < tolerance)
    assert np.array_equal(front | back | wall, lens_hit)
    assert np.all(radius[front | back] >= ring["inner_m"] - tolerance)
    return {"upstream_face": front, "bore_wall": wall, "downstream_face": back}


def entrance(voltage, phase, original, end, weights, mask, ring, field_reader, magnetic):
    field, receipt = field_reader("anode", True, voltage)
    surfaces_m = [[q[k] for k in ("start_m", "stop_m", "inner_m", "outer_m")] for q in receipt["stops"]]
    z = ring["start_m"]
    # A terminal observer after the existing physical surfaces in the list.
    # It only records first arrival; it contributes no field. Real front-face
    # impacts at that plane keep priority. The baseline's final hits are retained.
    observer = len(surfaces_m)
    surfaces_m.append([z, z, 0., max(row[3] for row in surfaces_m)])
    events, state, _ = transport(phase, field, magnetic, np.array(surfaces_m),
                                 np.empty(0), 0, .5)
    assert np.all(events >= -2)
    arriving = (events == observer) | mask["upstream_face"]
    np.testing.assert_array_equal(state[~arriving], end[~arriving])
    np.testing.assert_allclose(state[arriving, 2], z, rtol=0., atol=2e-16)
    r = np.hypot(state[:, 0], state[:, 1])
    np.testing.assert_array_equal(arriving & (r >= ring["inner_m"]), mask["upstream_face"])
    assert np.all(arriving[mask["bore_wall"]]) and np.all(state[arriving, 5] > 0.)
    early = events != observer
    decoded = np.where(events < 0, -events - 1, events + 2)
    np.testing.assert_array_equal(decoded[early], original[early])
    wall_r = r[mask["bore_wall"]]
    wall_state = state[mask["bore_wall"]]
    radial_u = (wall_state[:, :2] * wall_state[:, 3:5]).sum(axis=1) / wall_r
    inclinations = np.rad2deg(np.arctan2(radial_u, wall_state[:,5]))
    summary = dict(arriving_count=int(arriving.sum()), arriving_uA=float(weights[arriving].sum()),
                   maximum_arrival_radius_mm=1000*r[arriving].max(),
                   bore_wall_group_entry_radius_mm=(1000*np.array([wall_r.min(),wall_r.max()])).tolist(),
                   bore_wall_group_outward_angle_deg=[inclinations.min(), inclinations.max()])
    return summary, r


def plot(comparison, ids, example, ring):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.95), sharey=True)
    fig.subplots_adjust(left=.069, right=.98, bottom=.16, top=.78, wspace=.09)
    fig.suptitle("Which surface does the same electron strike?",x=.069,y=.98,ha="left",
                 fontsize=16.5,fontweight="bold",color="#192e43")
    fig.text(.069,.864,"Seven matched source particles. The highlighted trajectory has the same launch position and momentum in both panels.",
             fontsize=9.,color="#43586b")
    near, far, ir = (ring[k]*1000 for k in ("start_m","stop_m","inner_m"))
    for ax, (v, data), color in zip(axes, comparison.items(), ("#b15a27","#694b97")):
        ax.add_patch(Rectangle((near,ir),far-near,1.1,facecolor="#e0e5ea",edgecolor="#84919f",lw=1.))
        ax.axvline(near,ymin=0.,ymax=(ir-2.05)/(5.03-2.05),color="#65788c",ls=":",lw=1.)
        ax.text(near+.13,4.89,"Entry face",va="top",color="#354456",fontsize=9.)
        ax.text(19.85,4.08,"Bore wall",color="#354456",fontsize=9.)
        for i in sorted(ids,key=lambda i: i==example):
            zz = data["z_mm"]
            rr = data["path_r_mm"][i]
            hit_z,hit_r = data["end_zr_mm"][i]
            valid = np.isfinite(rr) & (zz <= hit_z)
            chosen = i==example
            ax.plot(np.r_[zz[valid],hit_z],np.r_[rr[valid],hit_r],
                    color=color,alpha=1 if chosen else .46,lw=2.4 if chosen else 1.)
            ax.scatter(hit_z,hit_r,s=31 if chosen else 10,color=color,zorder=4)
        if v==1200:
            ax.scatter(near,data["entry_r_mm"][example],s=41,facecolors="white",edgecolors=color,zorder=5)
        ax.set(xlim=(10.3,22.15),ylim=(2.05,5.03),xlabel="Distance from tip, z (mm)")
        ax.set_xticks([11,12,14,16,18,20,22])
        ax.set_title(f"+{v/1000:.2f} kV",loc="left",fontsize=11,fontweight="bold")
        ax.grid(axis="y",alpha=.17)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Distance from axis, r (mm)")
    fig.savefig(HERE/"gun-lens-impact-faces.png",dpi=180)
    plt.close(fig)


def run():
    set_num_threads(min(get_num_threads(), 14))
    reader = study_field_reader()
    with np.load(HERE/"gun-volume-field.npz") as mag:
        magnetic = np.ascontiguousarray(CubicSpline(mag["z_m"], mag["coeff"], axis=0).c.transpose(1,2,3,0))
    report, comparison = {}, {}
    for v in (50, 500, 1000, 1200, 1500, 1800, 2000):
        case = key("anode", True, v)
        receipt = json.loads((HERE/(case+".json")).read_text())
        ring, = [q for q in receipt["stops"] if q["key"]=="electrostatic_lens"]
        with np.load(WORK/(case+"-step0.5-rays.npz")) as data:
            end, original, weights = data["end"], data["label"], data["weight_uA"]
            hit = data["labels"][original]=="electrostatic_lens"
            mask = surfaces(end, hit, ring)
            row = {part: dict(count=int(selected.sum()),current_uA=float(weights[selected].sum()))
                   for part,selected in mask.items()}
            wall_z = end[mask["bore_wall"],2] * 1000
            row["first_bore_wall_strike_z_range_mm"] = [wall_z.min(), wall_z.max()]
            if v in (50,1200):
                phase=data["phase"]
                if v==50:
                    reference_phase,reference_weights,reference_ids=phase,weights,data["ids"]
                else:
                    np.testing.assert_array_equal(phase,reference_phase)
                    np.testing.assert_array_equal(weights,reference_weights)
                    np.testing.assert_array_equal(data["ids"],reference_ids)
                row["entrance"], r = entrance(v, phase, original, end, weights, mask, ring, reader, magnetic)
                comparison[v] = dict(z_mm=1000*data["z_plot"],path_r_mm=1000*np.linalg.norm(data["track_xy"],axis=2),
                                     end_zr_mm=1000*np.column_stack((end[:,2],np.hypot(end[:,0],end[:,1]))),
                                     mask=mask,entry_r_mm=1000*r)
            report[v]=row
    low,high=comparison.values()
    matched=np.flatnonzero(low["mask"]["upstream_face"] & high["mask"]["bore_wall"])
    saved=matched[matched < min(len(low["path_r_mm"]),len(high["path_r_mm"]))]
    ordered=saved[np.argsort(low["entry_r_mm"][saved])]
    ids=ordered[np.linspace(0,len(ordered)-1,7).round().astype(int)]
    example=int(ids[len(ids)//2])
    report["matched_face_to_wall"] = dict(count=len(matched),current_uA=float(reference_weights[matched].sum()),
        plotted_ids=ids.tolist(),example_id=example,
        example_entry_r_mm={v:row["entry_r_mm"][example] for v,row in comparison.items()},
        example_first_hit_zr_mm={v:row["end_zr_mm"][example].tolist() for v,row in comparison.items()})
    (HERE/"gun-lens-impact-faces.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    plot(comparison,ids,example,ring)
    print(json.dumps({k:report[k] for k in (50,1200,"matched_face_to_wall")},indent=2))


if __name__ == "__main__":
    run()

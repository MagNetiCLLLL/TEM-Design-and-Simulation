"""Compare the refocused baseline, small fitted Cs, and balanced physical working points."""

from hashlib import sha256
import json
from shutil import copyfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adjust_cs import CS_OUT, coefficients
from adjust_a2 import read_json
from fit_threefold import EVIDENCE, HERE, angle_groups, design, fit, grouped_current, read_sample, validation_folds
from plot_aperture_angle_spots import colour_bar, colour_scale, show_map
from plot_specimen_probe import landing_current_density
from temsim.physics.beam_statistics import transverse_beam_statistics


def independent_fits(s, omit=("C3", "C5")):
    basis, names = design(s)
    folds = validation_folds(s.ids)
    records = {}
    selections = {"full": np.ones(len(names), bool)}
    for term in omit:
        removed = np.isin(names, (term, term + "_x", term + "_y"))
        if not removed.any():
            raise ValueError(f"Unknown ray aberration {term}")
        selections["omit_" + term] = ~removed
    for key, keep in selections.items():
        predicted = np.empty_like(s.xy_um)
        m = basis[:,:,keep]
        for k in range(5):
            train = folds != k
            c, _ = fit(m[train],s.xy_um[train],s.current_na[train])
            predicted[~train] = m[~train] @ c
        records[key] = float(np.sqrt(np.average(np.sum((predicted-s.xy_um)**2,axis=1),weights=s.current_na)))
    return records


def stats(row, file, summary):
    s = read_sample(file,row,summary)
    radius = np.linalg.norm(s.xy_um,axis=1)
    w = s.current_na
    record = {k: row[k] for k in ("surface","through_focus","source_digest","emitted","survived","effective",
                                  "ring","current_ledger","common_waists_mm","waists_mm")}
    record.update(fit=coefficients(s),fit_inside_30mrad=coefficients(s,s.alpha_mrad<=30),
                  held_out_vector_RMS_um=independent_fits(s), angular_RMS_mrad=float(np.sqrt(np.average(s.alpha_mrad**2,weights=w))),
                  enclosed_fixed_axis_nA={str(r):float(w[radius<=r].sum()) for r in (.00005,.10,.25,.5,1.)})
    within = (radius>0)&(radius<=row["surface"]["D95_nm"]*.0005)
    unit = (s.xy_um[within,0]+1j*s.xy_um[within,1])/radius[within]
    record["H3_within_D95_radius"] = float(abs(np.average(unit**3,weights=w[within])))
    return s,record


def make_plot(samples, cases):
    labels = ("A₂ corrected and refocused", "C₃ near zero", "C₃ and C₅ balanced")
    windows = ((7.,.10,[-6,-3,0,3,6]), (1.,.025,[-1,-.5,0,.5,1]))
    maps = [[landing_current_density(s.xy_um,s.current_na,width,pitch)[0] for s in samples]
            for width,pitch,_ in windows]
    colours = colour_scale([g for row in maps for g in row])
    fig = plt.figure(figsize=(11.2,8.3),facecolor="white")
    fig.text(.071,.956,"Spherical-aberration tuning at the specimen",va="top",fontsize=19,fontweight="bold",color="#17212c")
    fig.text(.071,.904,"300 keV   ·   100 µm C2 aperture   ·   32,768 original source particles   ·   linear field interpolation",
             va="top",fontsize=10.5,color="#465362")
    grid = fig.add_gridspec(2,4,width_ratios=(1,1,1,.045),left=.071,right=.91,top=.783,bottom=.135,wspace=.19,hspace=.22)
    for i,(width,_,ticks) in enumerate(windows):
        for j, (case,label) in enumerate(zip(cases.values(),labels)):
            ax=fig.add_subplot(grid[i,j])
            img=show_map(ax,maps[i][j],width,ticks,colours)
            if i==0:
                ax.set_title(f"{label}\nD95 {case['surface']['D95_nm']*.001:.3f} µm",
                             fontsize=10.6,linespacing=1.5,pad=8)
            else:
                ax.set_xlabel("x (µm)")
            if j==0:
                ax.set_ylabel("y (µm)")
    colour_bar(fig,img,fig.add_subplot(grid[:,3]))
    fig.text(.071,.045,"Top  ±7 µm.     Bottom  ±1 µm.     Native specimen landings share the same optical axis and absolute colour scale.",
             fontsize=9.7,color="#465362")
    fig.savefig(HERE/"spherical-probe.png",dpi=175)
    plt.close(fig)


def final_probe_detail(sample, case, *, filename="spherical-probe-detail.png", sampling="linear, 0.30 mm field grid"):
    views = [(2., .025, [-2,-1,0,1,2]), (.5, .0125, [-.5,-.25,0,.25,.5])]
    maps = [landing_current_density(sample.xy_um, sample.current_na, width, pitch)[0]
            for width, pitch, _ in views]
    colours = colour_scale(maps)
    fig = plt.figure(figsize=(9.6,5.6), facecolor="white")
    fig.text(.079,.948,"Specimen probe after C₃–C₅ balancing",fontsize=19,fontweight="bold",
             color="#17212c",va="top")
    surface = case["surface"]
    fig.text(.079,.865, f"D95  {surface['D95_nm']:.0f} nm     ·     α95  {surface['alpha95_mrad']:.2f} mrad"
             f"     ·     {sampling}",fontsize=11,color="#465362",va="top")
    grid = fig.add_gridspec(1,3,width_ratios=(1,1,.047),left=.079,right=.896,
                            top=.708,bottom=.182,wspace=.32)
    for index, (values, (width, pitch, ticks), title) in enumerate(zip(maps, views, ("Outer distribution","Central spot"))):
        ax = fig.add_subplot(grid[0,index])
        image = show_map(ax, values, width, ticks, colours)
        ax.set(xlabel="x (µm)", ylabel="y (µm)")
        ax.set_title(f"{title}  ·  {pitch*1000:g} nm bins", fontsize=10.5, pad=7)
    colour_bar(fig,image,fig.add_subplot(grid[0,2]))
    fig.text(.079,.054,"Both views use the optical axis as origin and the same logarithmic current-density scale.",
             fontsize=9.5,color="#465362")
    fig.savefig(HERE/filename,dpi=185)
    plt.close(fig)


def angular_diagnostics(samples):
    final = samples[-1]
    groups = angle_groups(final)
    window = 1.
    maps = [landing_current_density(final.xy_um[g],final.current_na[g],window,.025)[0] for g in groups]
    colours = colour_scale(maps)
    fig=plt.figure(figsize=(12.2,6.7),facecolor="white")
    fig.text(.067,.957,"Which angles form the balanced probe?",fontsize=18,fontweight="bold",va="top",color="#17212c")
    fig.text(.067,.895,"Specimen upper surface  ·  identical ±1 µm windows  ·  measured mechanical angle about the chief ray",
             fontsize=10.5,va="top",color="#465362")
    top=fig.add_gridspec(1,5,left=.066,right=.915,bottom=.478,top=.78,wspace=.225,width_ratios=(1,1,1,1,.045))
    for i,(g,img,label) in enumerate(zip(groups,maps,("0–10","10–20","20–30","≥30"))):
        ax=fig.add_subplot(top[0,i])
        im=show_map(ax,img,window,[-1,0,1],colours)
        ax.set_title(f"{label} mrad  ·  {final.current_na[g].sum()*.001:.2f} µA",fontsize=10)
        ax.set_xlabel("x (µm)")
        if not i:
            ax.set_ylabel("y (µm)")
    colour_bar(fig,im,fig.add_subplot(top[0,4]))
    lower=fig.add_gridspec(1,2,left=.079,right=.952,bottom=.117,top=.334,wspace=.30)
    ax,enc=[fig.add_subplot(lower[0,k]) for k in range(2)]
    for s,label,color in zip(samples,("A₂ + focus","Small Cs","Balanced"),("#90613f","#495993","#157a70")):
        alpha=s.alpha_mrad
        edges=np.arange(0,42,1.)
        # Project the actual centroid-relative intercept onto its incident radial direction.
        uv=np.linalg.solve(s.frame_xy,s.angles_rad.T).T
        uv/=np.maximum(np.linalg.norm(uv,axis=1)[:,None],1e-20)
        xy=s.xy_um-np.average(s.xy_um,axis=0,weights=s.current_na)
        radial=np.sum(xy*uv,axis=1)
        w,_=np.histogram(alpha,edges,weights=s.current_na)
        first,_=np.histogram(alpha,edges,weights=s.current_na*radial)
        mask=(w>0)&(np.histogram(alpha,edges)[0]>=15)
        ax.plot(.5*(edges[1:]+edges[:-1])[mask],first[mask]/w[mask],lw=1.5,color=color,label=label)
        r=np.linalg.norm(s.xy_um,axis=1)
        sort=np.argsort(r)
        enc.plot(r[sort],np.cumsum(s.current_na[sort])*.001,lw=1.5,color=color,label=label)
    ax.axhline(0,color="#7e8992",ls=":",lw=.8)
    ax.set(xlabel="Incident semi-angle (mrad)",ylabel="Mean radial intercept (µm)",xlim=(0,40))
    enc.set(xscale="log",xlim=(.01,25),ylim=(0,16),xlabel="Radius about optical axis (µm)",ylabel="Enclosed current (µA)")
    ax.legend(frameon=False,fontsize=8,ncol=3,loc="upper left",labelspacing=.2,handlelength=1.3)
    for panel in (ax,enc):
        panel.spines[["top","right"]].set_visible(False)
        panel.tick_params(labelsize=9)
    fig.savefig(HERE/"spherical-angle-response.png",dpi=175)
    plt.close(fig)


def main():
    ref=read_json(EVIDENCE/"refocused-working-point.json")
    summary=read_json(EVIDENCE/"comparison.json")
    cases={}
    samples=[]
    files=[]
    for name in ("baseline","zero-native","balanced-native"):
        if name=="baseline":
            row=ref["paired_controls"]["refocused"]["after"]
            path=EVIDENCE/ref["trajectory_file"]
        else:
            row=read_json(CS_OUT/(name+".json"))
            path=CS_OUT/row["phase_file"]
        assert sha256(path.read_bytes()).hexdigest()==row["trajectory_sha256"]
        s,r=stats(row,path,summary)
        if name!="baseline":
            r.update({k:row[k] for k in ("tl12_gain","mini_gain","objective_At","currents_A","corrector_sha256",
                                          "trajectory_sha256","corrector_field","objective_solution","C2_foil")})
            assert row["source_digest"]==ref["paired_controls"]["refocused"]["after"]["source_digest"]
            assert row["objective_solution"]["installed_geometry_digest"]==ref["objective_solution"]["installed_geometry_digest"]
            accepted=EVIDENCE/("cs-"+name+".npz")
            copyfile(path,accepted)
            r["trajectory_file"]=accepted.name
        else:
            r.update(trajectory_file=path.name,trajectory_sha256=row["trajectory_sha256"],objective_At=ref["objective_At"],
                     tl12_gain=1.,mini_gain=1.,currents_A=ref["currents_A"])
        files.append(path)
        samples.append(s)
        cases[name]=r

    with np.load(files[0]) as a, np.load(files[-1]) as b:
        for key in ("ids","weights","energy_offset_eV"):
            np.testing.assert_array_equal(a[key],b[key])
        live=(a["loss"]==-1)&(b["loss"]==-1)
        common={"count":int(live.sum()),"current_uA":float(a["weights"][live].sum()*ref["gun"]["gun_exit_A"]*1e6)}
        for name,data in (("baseline",a),("balanced",b)):
            p=data["phase"][live,:4]
            m=transverse_beam_statistics(*p.T, weights=data["weights"][live])
            common[name+"_D95_um"]=float(2e6*m.radius_95_m)
    controls={}
    for k in ("balanced-halfstep","balanced-prefix","balanced-fine-xy","balanced-mesh5"):
        j=read_json(CS_OUT/(k+".json"))
        r=cases["balanced-native"]
        for key in ("objective_At","tl12_gain","mini_gain","currents_A"):
            np.testing.assert_array_equal(j[key],r[key])
        assert j["corrector_field"]["currents_A"]==r["corrector_field"]["currents_A"]
        controls[k]={n:j[n] for n in ("emitted","survived","surface","through_focus","fit","ring","corrector_sha256","max_step_mm")}
    drive_checks={}
    for name in ("cs-base","cs-hex3","cs-hex0","cs-transfer6"):
        j=read_json(CS_OUT/(name+".json"))
        drive_checks[name]=dict(corrector_currents_A=j["corrector_currents_A"],**j["trials"][0])
    sweep=[]
    for name in ("relay-zero","relay-balance"):
        for j in read_json(CS_OUT/(name+"-selected.json")):
            sweep.append({k:j[k] for k in ("target_Cs_mm","objective_At","tl12_gain","mini_gain","surface","fit","drift_D95_um","approximate_objective")})
    brackets = []
    for name in ("bracket-low", "balance300", "bracket-high"):
        trial = read_json(CS_OUT/(name+"-selected.json"))[-1]
        brackets.append({k:trial[k] for k in ("objective_At","tl12_gain","mini_gain","surface","fit","drift_D95_um","approximate_objective")})
    trim = []
    for name in ("balanced-pretrim","trim-response-xy","trim-response-y"):
        item = read_json(CS_OUT/(name+".json"))
        trim.append(dict(currents_A=item["corrector_currents_A"],fit=item["trials"][0]["fit"],
                         surface=item["trials"][0]["surface"]))
    input_pairs = np.array([[item["currents_A"]["probe_hpol_hexapole_"+axis] for axis in ("x","y")] for item in trim])
    quadratures = np.array([item["fit"]["A2"]["cos_sin_mm"] for item in trim])
    response = np.linalg.solve(input_pairs[1:]-input_pairs[0],quadratures[1:]-quadratures[0]).T
    result=dict(energy_keV=ref["energy_keV"],gun=ref["gun"],C2_aperture_diameter_um=ref["C2_aperture_diameter_um"],
                specimen_upper_surface_z_m=ref["specimen_upper_surface_z_m"],owner_per_group=ref["owner_per_group"],
                objective_turns=ref["objective_turns"],cases=cases,controls=controls,common_survivors=common,
                fixed_focus_corrector_trials=drive_checks,exploratory_sweep=sweep,exact_brackets_before_trim=brackets,
                fine_trim=dict(trials=trim,A2_response_mm_per_A=response.tolist()),
                balanced_angle_groups=grouped_current(samples[-1]))
    (EVIDENCE/"spherical-working-points.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    make_plot(samples,cases)
    final_probe_detail(samples[-1],cases["balanced-native"])
    angular_diagnostics(samples)


if __name__=="__main__":
    main()

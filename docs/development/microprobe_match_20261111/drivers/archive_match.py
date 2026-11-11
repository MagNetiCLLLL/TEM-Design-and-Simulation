"""Archive completed classical Microprobe conjugacy measurements."""
from pathlib import Path
from hashlib import sha256
import gzip
import json
import shutil
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DEST=ROOT/"docs/development/microprobe_match_20261111"
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
import numpy as np
import tomli_w
from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.profile_io import apply_profile_values,read_profile,save_profile


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf8")


def signature(path):
    return dict(sha256=sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


def main():
    runs={name:read(HERE/"column"/name/"report.json") for name in
          ("n1024-first","n1024-step010-v2","n2048-step010","n2048-endpoints-v2",
           "n2048-high-hp010-v2","n2048-hp010-family","n2048-final-hp010-endpoints",
           "n1024-hp010-nominal")}
    if any(row["status"]!="COMPLETE" for row in runs.values()):
        raise ValueError("All physical batches must complete first")
    if DEST.exists():
        raise FileExistsError("Historical archives are never overwritten")
    DEST.mkdir(parents=True)
    manifest=dict(scope="Executed source, field and current evidence with retained native observations",
        source_cache="outputs/microprobe-20261111/extractor/new-n2048",
        source_archive="../microprobe_20261111/extractor/new-n2048",files={},original_full_files={})

    def copy(source,target,compressed=False):
        dest=DEST/target
        dest.parent.mkdir(parents=True,exist_ok=True)
        if compressed:
            dest.write_bytes(gzip.compress(source.read_bytes(),mtime=0))
        else:
            shutil.copyfile(source,dest)
        manifest["files"][target]=dict(original_path=str(source.relative_to(ROOT)).replace("\\","/"),
            original=signature(source),**signature(dest))

    for name in runs:
        folder=HERE/"column"/name
        copy(folder/"report.json",f"column/{name}/report.json")
        for case in sorted(folder.glob("trial-??")):
            if not (case/"report.json").is_file():
                continue
            target=f"column/{name}/{case.name}"
            copy(case/"report.json",f"{target}/report.json.gz",compressed=True)
            copy(case/"inputs.json",f"{target}/inputs.json.gz",compressed=True)
            copy(case/"input.toml",f"{target}/input.toml")
            for filename in ("waist-audit.json","rotation-analysis.json","beam-envelope.npz"):
                if (case/filename).exists():
                    copy(case/filename,f"{target}/{filename}")
            manifest["original_full_files"][str((case/"observations.npz").relative_to(ROOT)).replace("\\","/")]=signature(case/"observations.npz")
    selection=[("0p75um","n2048-final-hp010-endpoints",0),("0p97um","n2048-hp010-family",1),
               ("1p49um","n2048-hp010-family",2),("2p19um","n2048-final-hp010-endpoints",1)]
    source_doc=tomllib.loads((ROOT/"outputs/microprobe-20261111/extractor/new-n2048/input.toml").read_text())
    results=dict(scope="Finite-field Microprobe core conjugacy and parallel illumination at reported sampling",
        reference_plane_mm=1599.1999998000001,source_positive_count=2048,source_current_a=1e-5,
        family=[],sample_scattering="NOT_RUN",coherent_propagation="NOT_RUN",global_preset_changed=False,
        ceos_oem_equivalence="NOT_CLAIMED",complete_aberration_coefficients="NOT_QUALIFIED")
    rawkeys=("x_m","y_m","tx_rad","ty_rad","kinetic_energy_ev","flight_time_s","alive","z_mm")
    for slug,name,index in selection:
        case=HERE/"column"/name/f"trial-{index:02d}"
        row=read(case/"report.json")
        audit=read(case/"waist-audit.json")
        surface=row["surface"]
        if not .75<=surface["d95_um"]<=2.2 or surface["alpha95_mrad"]>.3:
            raise ValueError("Measured family point is outside the requested diameter/angle domain")
        if audit["actual_waist_count"]!=4:
            raise ValueError("Unexpected full-column source-image topology")
        if any(value["unresolved"] for populations in row["topology"].values() for value in populations.values()):
            raise ValueError("Unresolved finite-source waist interval")
        document=tomllib.loads((case/"input.toml").read_text())
        document["devices"]["sample"]=source_doc["devices"]["sample"].copy()
        document["sample_model"]=source_doc["sample_model"].copy()
        filename=f"microprobe-matched-{slug}-si110.toml"
        path=DEST/filename
        path.write_text("# Measured Microprobe conjugate transport and parallel-illumination candidate.\n"
            "# No OEM equivalence, full Cs correction, or scanning-image qualification is asserted.\n"
            "# Original Si specimen and absolute local CIF path restored; see README.md.\n"+
            tomli_w.dumps(document),encoding="utf8")
        assembled,values=read_profile(path)
        state=default_state();catalog=AssemblyCatalog()
        catalog.apply(state,catalog.normalise_selection(assembled));apply_profile_values(state,values)
        if (state.mini_condenser.polarity!=1 or state.illumination_mode!="TEM" or state.step_mm!=.1
                or state.electron_gun.emitter.ray_count!=2048
                or abs(state.sample.upper_surface_z_mm-surface["z_mm"])>1e-10):
            raise ValueError("Saved candidate did not restore source, mode or physical surface")
        for part in (*state.lenses,*state.corrector_elements):
            for field,value in row["controls"].get(part.key,{}).items():
                if not np.isclose(getattr(part,field),value,rtol=0.,atol=1e-12):
                    raise ValueError("Profile changed a recorded optical drive")
        keep_planes=[entry["z_mm"] for entry in row["axial_profile"]]
        for waist in audit["waists"]:
            for field in ("rms_native_bracket_mm","d95_native_bracket_mm"):
                keep_planes.extend(waist[field])
        with np.load(case/"observations.npz",allow_pickle=False) as data:
            planes=np.unique(keep_planes)
            indices=np.searchsorted(data["z_mm"],planes)
            if not np.array_equal(data["z_mm"][indices],planes):
                raise ValueError("Retained source-waist planes must be actual observations")
            retained={key:data[key][indices] if key in rawkeys else data[key] for key in data.files}
            dest=DEST/f"column/{name}/{case.name}/native-selected-phase.npz"
            np.savez_compressed(dest,**retained)
        results["family"].append(dict(profile=filename,run=name,index=index,surface=surface,
            controls=row["controls"],waists=audit["waists"],ordered_nearest_component_intervals=audit["ordered_nearest_component_intervals"],
            per_stage_losses=row["per_stage_losses"],absolute_current_balance_residual_a=row["absolute_current_balance_residual_a"],
            max_radius_to_r95=audit["max_radius_to_r95"],max_angle_to_alpha95=audit["max_angle_to_alpha95"],
            input_digest=row["execution"]["input_digest"],source_identity=row["execution"]["source_identity_sha256"],
            original_positive_count=row["execution"]["emitted_positive_current_points"],
            profile_roundtrip="PASS; original specimen settings restored; no scattering added"))
    for filename in ("native-maps.json","native-maps.npz","refined-conjugates.json","comparisons.json","final-comparisons.json",
                     "research-sources.json","gun-waist-diagnostic.json"):
        if (HERE/filename).exists():
            copy(HERE/filename,filename)
    for source in sorted(HERE.glob("*.py")):
        copy(source,"drivers/"+source.name)
    for source in sorted((HERE/"column").glob("*.py")):
        copy(source,"drivers/column/"+source.name)
    write(DEST/"results.json",results)
    write(DEST/"microprobe-contract.json",dict(scope="This measured fixed-geometry/source family only",
        requested_d95_um=[.75,2.2],maximum_alpha95_mrad=.3,source_waist_definition="Executed current-weighted two-dimensional RMS region; D95 minima separate",
        source_absolute_interval_mm=[450.,1599.1999998000001],source_waist_count=4,
        source_image_intervals=["C1-C2","C2-C3","TL22-TL21 (between DP22 and DP21)","Mini Condenser-Upper Objective"],
        ac_interval="No static source-image waist required or observed in this Microprobe family",
        historical_six_waist_reference="Retained as historical data; not a qualification gate for this new family",
        architecture="Literature-guided model design, not a published CEOS Microprobe current table",
        profile_hashes={entry["profile"]:signature(DEST/entry["profile"]) for entry in results["family"]}))
    for path in DEST.rglob("*"):
        if path.is_file():
            key=str(path.relative_to(DEST)).replace("\\","/")
            manifest["files"].setdefault(key,dict(generated_from="Complete immutable physical observations",**signature(path)))
    write(DEST/"artifact_manifest.json",manifest)
    print(json.dumps(dict(archive=str(DEST),family=[dict(profile=row["profile"],d95_um=row["surface"]["d95_um"],
        alpha95_mrad=row["surface"]["alpha95_mrad"]) for row in results["family"]]),indent=2),flush=True)


if __name__=="__main__":
    main()

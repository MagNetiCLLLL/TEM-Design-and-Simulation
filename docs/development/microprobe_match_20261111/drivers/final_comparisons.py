"""Separate multipole response, count sensitivity and historical step controls."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DEST=ROOT/"docs/development/microprobe_match_20261111"


def read(run,index=0):
    return json.loads((HERE/"column"/run/f"trial-{index:02d}"/"report.json").read_text())


def compare(before,after):
    return {key:dict(before=before["surface"][key],after=after["surface"][key],
                relative_change=after["surface"][key]/before["surface"][key]-1)
            for key in ("d95_um","radial_rms_um","alpha95_mrad","alpha_rms_mrad","current_a")}


def unresolved(row):
    return {key:value["actual_population"]["unresolved"] for key,value in row["topology"].items()
            if value["actual_population"]["unresolved"]}


def main():
    old=read("n2048-endpoints-v2",1);reduced=read("n2048-high-hp010-v2")
    small=read("n1024-hp010-nominal");large=read("n2048-hp010-family",1)
    changed=[key for key,value in old["controls"].items() if value!=reduced["controls"][key]]
    assert set(changed)=={"probe_hp1_hexapole","probe_hp2_hexapole","probe_hpc_hexapole","probe_hpol_hexapole"}
    for key in changed:
        for field,value in old["controls"][key].items():
            expected=value*.1 if field in {"strength_m3","skew_strength_m3"} else value
            assert reduced["controls"][key][field]==expected
    assert old["execution"]["source_identity_sha256"]==reduced["execution"]["source_identity_sha256"]
    assert small["execution"]["emitted_positive_current_points"]==1024
    assert large["execution"]["emitted_positive_current_points"]==2048
    result=dict(scope="Physical controls, with no restored clipping or altered source current",
        hexapole_amplitude_control=dict(original_source_count=2048,fraction_of_inherited_amplitudes=.1,
            same_upstream_controls=True,same_current=old["surface"]["current_a"]==reduced["surface"]["current_a"],
            before_unresolved=unresolved(old),after_unresolved=unresolved(reduced),metrics=compare(old,reduced)),
        final_nominal_count_control=dict(before_count=1024,after_count=2048,step_mm=.1,
            same_optical_drives=small["controls"]==large["controls"],metrics=compare(small,large)),
        step_scope="The separate comparisons.json integration refinement belongs to inherited hexapole drives, not a new final-family step test",
        final_full_aberration_qualification="NOT_RUN",final_full_raster_qualification="NOT_RUN")
    assert not result["hexapole_amplitude_control"]["after_unresolved"]
    assert result["final_nominal_count_control"]["same_optical_drives"]
    (HERE/"final-comparisons.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    if DEST.exists() and (DEST/"results.json").exists():
        (DEST/"final-comparisons.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()

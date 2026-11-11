"""Independent NumPy verification of retained absolute-current phase space."""
from pathlib import Path
from hashlib import sha256
import json
import math
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/"docs/development/microprobe_match_20261111"
SOURCE=ROOT/"docs/development/microprobe_20261111/extractor/new-n2048"


def quantile(value,weight):
    order=np.argsort(value)
    threshold=np.nextafter(.95*math.fsum(weight),-np.inf)
    for i in range(len(order)):
        if math.fsum(weight[order[:i+1]])>=threshold:
            return float(value[order[i]])
    raise ValueError("Unreached containment current")


def main():
    results=json.loads((DEST/"results.json").read_text())
    with np.load(SOURCE/"gun_exit_and_emission.npz") as src:
        source_ids=src["source_id"].copy();weights=src["weights"].copy()
        exit_phase={key:src["gun_exit_"+key].copy() for key in ("x_m","y_m","tx_rad","ty_rad")}
        exit_phase["kinetic_energy_ev"]=300000.+src["gun_exit_energy_offset_ev"]
        exit_phase["flight_time_s"]=src["gun_exit_flight_time_s"].copy()
    checked=[]
    for entry in results["family"]:
        path=DEST/"column"/entry["run"]/f"trial-{entry['index']:02d}"/"native-selected-phase.npz"
        with np.load(path,allow_pickle=False) as data:
            assert np.array_equal(data["source_id"],source_ids)
            assert np.array_equal(data["weights"],weights)
            assert np.count_nonzero(weights>0)==2048 and len(np.unique(source_ids))==2048
            with np.load(path.parent/"native-gun-exit.npz",allow_pickle=False) as initial:
                assert float(initial["z_mm"])==450.
                assert np.array_equal(initial["source_id"],source_ids)
                assert np.array_equal(initial["weights"],weights)
                energy_error=np.abs(initial["kinetic_energy_ev"]-exit_phase["kinetic_energy_ev"])
                assert np.all(energy_error<=np.spacing(exit_phase["kinetic_energy_ev"]))
                for key,value in exit_phase.items():
                    if key!="kinetic_energy_ev":assert np.array_equal(initial[key],value),key
            row=int(np.flatnonzero(data["z_mm"]==entry["surface"]["z_mm"])[0])
            keep=data["alive"][row]&(weights>0)
            assert np.array_equal(keep,data["terminal_alive"])
            w=weights[keep]
            xy=np.c_[data["x_m"][row,keep],data["y_m"][row,keep]]
            r=np.linalg.norm(xy-np.average(xy,axis=0,weights=w),axis=1)
            v=np.c_[data["tx_rad"][row,keep],data["ty_rad"][row,keep],np.ones(keep.sum())]
            v/=np.linalg.norm(v,axis=1)[:,None]
            chief=np.average(v,axis=0,weights=w);chief/=np.linalg.norm(chief)
            angle=np.arctan2(np.linalg.norm(np.cross(v,chief),axis=1),v@chief)
            measured=dict(d95_um=quantile(r,w)*2e6,radial_rms_um=np.sqrt(np.average(r*r,weights=w))*1e6,
                alpha95_mrad=quantile(angle,w)*1e3,alpha_rms_mrad=np.sqrt(np.average(angle*angle,weights=w))*1e3,
                count=int(keep.sum()),effective_count=w.sum()**2/np.dot(w,w),current_a=1e-5*w.sum())
            for key,value in measured.items():
                # The library reports angular RMS in a small-angle slope chart.
                tol=1e-7 if key=="alpha_rms_mrad" else 2e-12
                assert np.isclose(value,entry["surface"][key],rtol=tol,atol=1e-15),(key,value,entry["surface"][key])
            assert len([x for x in entry["waists"] if x["population"]=="actual_population"])==4
            checked.append(dict(profile=entry["profile"],source_ids_and_absolute_weights_equal=True,
                gun_exit_transverse_phase_and_clock_bitwise_equal=True,
                gun_exit_maximum_energy_roundtrip_error_ev=float(energy_error.max()),
                gun_exit_energy_roundtrip_within_one_ulp=True,
                terminal_interceptions_preserved=True,metrics=measured,phase_sha256=sha256(path.read_bytes()).hexdigest()))
    out=dict(status="VERIFIED",scope="Independent retained-array verification; no propagation, clipping or retuning",profiles=checked)
    (DEST/"phase-verification.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf8")
    print(json.dumps(out,indent=2))


if __name__=="__main__":main()

"""Full incident-column Microprobe survey using the verified source driver."""
from pathlib import Path
from hashlib import sha256
import importlib.util
import json
import math
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
LEGACY=ROOT/"outputs/microprobe-20261111/column/run_microprobe.py"
spec=importlib.util.spec_from_file_location("micro_full_source_driver",LEGACY)
driver=importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def full_planes(state, spacing):
    surface=float(state.sample.upper_surface_z_mm)
    nearby=np.unique(np.r_[surface+np.array((-100.,-10.,-1.,0.,1.,10.,100.))*.001,
                          state.sample.z_mm,state.sample.lower_surface_z_mm])
    parts={part.key:part for part in (*state.lenses,*state.corrector_elements)}
    names=("condenser_lens_1","condenser_lens_2","condenser_lens_3","adapter_lens",
           "probe_hp2_hexapole","probe_tl22_lens","probe_tl21_lens","probe_hp1_hexapole","probe_tl12_lens")
    chain=[("gun_exit",float(state.electron_gun.exit_plane_z_mm))]
    chain.extend((name,float(parts[name].z_mm)) for name in names)
    chain.extend((("ac_upper",float(state.ac_deflector.upper_z_mm)),
                  ("ac_lower",float(state.ac_deflector.lower_z_mm)),
                  ("mini_condenser",float(state.mini_condenser.z_mm)),("specimen_entrance",surface)))
    intervals=[]
    rows=[]
    for (a,lower),(b,upper) in zip(chain[:-1],chain[1:]):
        step=min(spacing,.05 if a=="ac_upper" else .1 if a in ("probe_tl22_lens","mini_condenser") else spacing)
        intervals.append((a+" to "+b,lower,upper,step))
        rows.append(np.linspace(lower,upper,math.ceil((upper-lower)/step)+1))
    exact=[float(part.z_mm) for part in (*state.corrector_elements,*state.apertures,*state.stigmators)
           if part.enabled and chain[0][1]<=part.z_mm<=nearby[-1]]
    exact.extend((state.beam_deflector.upper_z_mm,state.beam_deflector.lower_z_mm,
                  state.condenser_deflector.optical_upper_reference_from_tip_mm,
                  state.condenser_deflector.optical_lower_reference_from_tip_mm,
                  state.objective_lens.upper_pole_piece_center_z_mm))
    return nearby,intervals,np.unique(np.r_[nearby,np.concatenate(rows),exact])


if __name__=="__main__":
    driver.observation_planes=full_planes
    if "--output" in sys.argv:
        out=Path(sys.argv[sys.argv.index("--output")+1])
        proof=out.parent/(out.name+"-wrapper.json")
        proof.parent.mkdir(parents=True,exist_ok=True)
        proof.write_text(json.dumps(dict(wrapper_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            reused_driver_sha256=sha256(LEGACY.read_bytes()).hexdigest(),
            scope="Full incident-column survey; no Nanoprobe count gate; actual original weights"),indent=2)+"\n",encoding="utf8")
    driver.main()

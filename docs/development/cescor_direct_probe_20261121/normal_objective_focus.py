"""Place a selected material-FEM corrector setting on the specimen with the objective supply."""

import argparse
from hashlib import sha256

import numpy as np

from downstream_pivot import inputs, write
from illuminated_observation import capture
from normal_corrector_search import field_path, fit_surface
from refocus_a2 import objective_receipt
from relay_transfer import Native, SURFACE, WORK


def focus(tag, step, count, initial, target_offset=0.0, decimals=1):
    ref, _ = inputs()
    q = ref["currents_A"]
    path = field_path(tag)
    model = Native(path, step, count, ref["objective_At"])
    traces = {}
    increment = 10.**(-decimals)
    def trial(ni):
        ni = round(float(ni), decimals)
        if ni not in traces:
            receipt = objective_receipt(ni)
            model.ni, model.bg = ni, model.m.base("armco", ni, "gap-fine", 8)
            _, ((p, loss),) = model.replay(q, [SURFACE])
            row = fit_surface(p, loss, model.t)
            traces[ni] = dict(objective_At=ni, field_digest=receipt["field_digest"], **row)
            print(tag, ni, "D95 um, waist offset um, 50 nm disk nA",
                  row["D95_um"], row["drift_D95_um"], row["axis_50nm_nA"], flush=True)
        return traces[ni]["drift_D95_um"] - target_offset
    ni = initial
    for _ in range(4):
        off = trial(ni)
        if abs(off) < 1.5*increment:
            break
        # Three micrometres per ampere-turn is the measured small-signal starting
        # slope on this branch. Subsequent corrections use the measured secant.
        slope = -3.1
        other = [x for x in traces if abs(x-ni) >= .99*increment]
        if other:
            earlier = min(other, key=lambda x: abs(x-ni))
            slope = (trial(ni)-trial(earlier))/(ni-earlier)
        if not -6 < slope < -1:
            raise ValueError(f"Objective response outside measured branch: {slope}")
        proposed = round(ni - off/slope, decimals)
        if proposed == ni:
            break
        ni = proposed
    center = min(traces, key=lambda x: abs(trial(x)))
    for x in (center-increment, center+increment):
        trial(x)
    ni = min(traces, key=lambda x: abs(trial(x)))
    data = dict(reference="illuminated-normal.json / final", corrector_tag=tag,
                corrector_sha256=sha256(path.read_bytes()).hexdigest(),
                winding_currents_A=q, gains=ref["gains"], objective_At=ni,
                objective_current_A=ni/1160, objective_turns=1160, emitted=count,
                target_offset_um=target_offset, step_mm=step, increment_At=increment,trials=list(traces.values()))
    write(WORK/(tag+"-focus.json"), data)
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tag")
    p.add_argument("--ni", type=float, default=-14551.5)
    p.add_argument("--count", type=int, default=32768)
    p.add_argument("--step", type=float, default=.0375)
    p.add_argument("--decimals", type=int, choices=(1,2,3,4), default=1)
    p.add_argument("--capture", action="store_true")
    p.add_argument("--fixed-ni", action="store_true", help="Use the supplied excitation for a fixed-control refinement")
    args = p.parse_args()
    if args.fixed_ni:
        ref, _ = inputs()
        label = args.tag+f"-fixed-{args.step:g}"
        write(WORK/(label+"-focus.json"), dict(gains=ref["gains"], objective_At=args.ni))
    else:
        label = args.tag
        focus(args.tag,args.step,args.count,args.ni,decimals=args.decimals)
    if args.capture:
        capture(label+"-focus.json", label+"-capture", args.count,args.step,field_path(args.tag), axis_core=True)

"""Replay the illuminated condenser branch, then place its native D95 waist."""

import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from condenser_normal import FIELD, FOIL, HERE, SURFACE, WORK, observation_planes, reference
from illuminated_normal import controls
from refocus_a2 import objective_receipt
from relay_transfer import Native


def focus_offsets_nm():
    return np.unique(np.r_[np.arange(-300000., 300001., 1000.), np.arange(-2000.,2001.,50.),
                           np.arange(-100.,101.,2.), np.arange(-4.,4.01,.25)])


def column_planes():
    return np.unique(np.r_[observation_planes(.25), SURFACE + focus_offsets_nm()*1e-9])


def output(name, data):
    (WORK / (name+".json")).write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf-8")


def candidates(path):
    if path.suffix == ".json":
        row = json.loads(path.read_text())
        return [(np.array(row["gains"]), row["objective_At"])]
    with np.load(path) as raw:
        g, s = raw["gains"], raw["result"]
    ok = (s[:,0]==1)&(s[:,1]==0)&(s[:,6]==1)&(s[:,7]==0)&(s[:,8]==0)&(s[:,5]>=1.05)
    indices = np.flatnonzero(ok)
    order = indices[np.argsort((np.maximum(0,1.75-s[indices,5])*20 + abs(s[indices,3]-30)
                               + .07*s[indices,4]))]
    ni=reference()["objective_At"]
    return [(g[i],ni) for i in order]


class Focus:
    def __init__(self, count, step, ni, full_grid=False):
        self.model = Native(FIELD, step, count, ni, column_planes() if full_grid else ())
        self.base = np.array(reference()["currents_A"])
        from illuminated_normal import winding_groups
        self.groups = winding_groups(True)
        self.journal = []

    @lru_cache(maxsize=12)
    def background(self, ni):
        objective_receipt(ni)
        return self.model.m.base("armco", ni, "gap-fine", 8)

    def run(self, gains, ni, *, details=False):
        q = self.base.copy()
        for group,gain in zip(self.groups,gains,strict=True):
            q[group] *= gain
        if self.model.ni != ni:
            self.model.bg = self.background(ni)
            self.model.ni = ni
        row, _, _ = self.model.measure(q, details=details)
        vals = row["rim"]["angle_mrad_min_median_max"]
        if row["rim"]["reached_sample"] != row["rim"]["azimuths"]:
            raise ValueError("Aperture rim did not survive the installed column")
        angle = vals[-1]
        brief = dict(gains=list(gains), objective_At=float(ni), edge_mrad=angle,
                     drift_D95_um=row["drift_D95_um"], D95_nm=row["specimen"]["D95_nm"],
                     foil_D95_um=row["foil_entrance_D95_um"], pupil_rim=row["entrance"]["beyond_90pct_count"],
                     emitted=row["emitted"], step_mm=row["max_step_mm"])
        self.journal.append(brief)
        print(f'{len(self.journal):2d} C3 {gains[-2]:.8f} Mini {gains[-1]:.10f} objective {ni:.1f} '
              f'edge {angle:.6f} drift {row["drift_D95_um"]:.6f} um D95 {brief["D95_nm"]:.4f} nm '
              f'foil D95 {brief["foil_D95_um"]:.2f} um rim n {brief["pupil_rim"]}', flush=True)
        return np.array([angle-30., row["drift_D95_um"]]), row


def tune(args):
    options = candidates(WORK / args.seed)
    if not options:
        raise ValueError("No illuminated single-crossover starting point")
    g, ni = options[args.index]
    g = g.copy()
    model = Focus(args.count, args.step, ni, args.full_grid)
    err, row = model.run(g, ni)
    def save():
        output(args.name,dict(gains=g.tolist(), objective_At=ni, row=row,journal=model.journal))
    save()
    if args.stage == "angle":
        # Fix illumination and both downstream focusing controls while setting the rim angle.
        for _ in range(7):
            if abs(err[0]) < .012:
                break
            trial=g.copy()
            trial[-2]+=.0001
            delta=model.run(trial,ni)[0][0]-err[0]
            if abs(delta)<1e-7:
                break
            g[-2]=np.clip(g[-2]+np.clip(-err[0]*.0001/delta,-.035,.035),.055,.49)
            err,row=model.run(g,ni)
            save()
        return
    # Set both physical focusing supplies; at the 0.1 At resolution of the
    # material solve, use the Mini Condenser to finish the D95 position.
    for iteration in range(10):
        if abs(err[0]) < .18 and abs(err[1]) < .00012:
            break
        dmini = .00001 if abs(err[1]) < 1.0 else .0001
        probe = g.copy()
        probe[-1] += dmini
        jmini = (model.run(probe,ni)[0]-err)/dmini
        if abs(err[0]) < .10 and abs(err[1]) < 2.0 and iteration > 0:
            delta = np.array([-err[1]/jmini[1],0.])
        else:
            dni = 10. if iteration < 2 else 1.
            jni = (model.run(g,round(ni+dni,1))[0]-err)/dni
            delta = np.linalg.lstsq(np.column_stack((jmini,jni)), -err,rcond=None)[0]
        delta *= min(1., .012/max(abs(delta[0]),1e-15),450./max(abs(delta[1]),1e-15))
        pre = np.linalg.norm(err/[.18,5.])
        for a in (1., .5, .16, .04):
            probe = g.copy()
            probe[-1] += a*delta[0]
            target = round(ni+a*delta[1],1)
            if not (.75 < probe[-1] < 1.25 and -36000 <= target <= -4500):
                continue
            trial, info = model.run(probe,target)
            if np.linalg.norm(trial/[.18,5.]) < pre or (abs(trial[0]) < .18 and abs(trial[1]) < abs(err[1])):
                g, ni, err, row = probe,target,trial,info
                save()
                break
        else:
            break


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("seed")
    parser.add_argument("name")
    parser.add_argument("--count",type=int,default=8192)
    parser.add_argument("--step",type=float,default=.1)
    parser.add_argument("--stage",choices=("angle","focus"), default="focus")
    parser.add_argument("--index",type=int,default=0)
    parser.add_argument("--full-grid",action="store_true")
    tune(parser.parse_args())

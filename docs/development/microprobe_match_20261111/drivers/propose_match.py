"""Axial-field first-order proposals, never a finite-source qualification."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
from numba import njit
from scipy.optimize import least_squares
from temsim.instrument_snapshot import InstrumentSnapshot
from temsim.physics.potential_capture import capture_round_lens_potential
from temsim.physics.round_lens_aberration_benchmark import _charge_over_momentum
from temsim.physics.beam_statistics import transverse_beam_statistics
from validate_joint_probe import _restore_exact

HERE = Path(__file__).resolve().parent
GUN = ROOT / "outputs/microprobe-20261111/extractor/new-n2048"


@njit(cache=True)
def matrices(fields, drives, lengths, marks, rigidity):
    result = np.empty((len(marks), 2, 2))
    a, b, c, d, j = 1., 0., 0., 1., 0
    for i in range(len(lengths)):
        bz = 0.
        for k in range(len(drives)):
            bz += fields[k, i] * drives[k]
        kappa = abs(rigidity * bz)
        sine = np.sin(kappa*lengths[i])/kappa if kappa else lengths[i]
        cosine = np.cos(kappa*lengths[i])
        kick = -kappa*kappa*sine
        a, b, c, d = cosine*a+sine*c, cosine*b+sine*d, kick*a+cosine*c, kick*b+cosine*d
        if j < len(marks) and i+1 == marks[j]:
            result[j, 0, 0], result[j, 0, 1], result[j, 1, 0], result[j, 1, 1] = a, b, c, d
            j += 1
    return result


class Proposal:
    def __init__(self):
        self.state = _restore_exact(InstrumentSnapshot.from_dict(json.loads((GUN / "inputs.json").read_text())))
        state = self.state
        self.lenses = [part for part in state.lenses if part.enabled and part.z_mm <= state.sample.z_mm]
        self.keys = [part.key for part in self.lenses]
        self.base = np.array([part.percent for part in self.lenses])
        self.maximum = np.array([part.max_percent for part in self.lenses])
        parts = {part.key: part for part in (*state.lenses, *state.corrector_elements)}
        self.planes = dict(c2ap=state.condenser_aperture_2.z_mm,
            hp2=parts["probe_hp2_hexapole"].z_mm, central=1211.6333,
            hp1=parts["probe_hp1_hexapole"].z_mm, tl12=parts["probe_tl12_lens"].z_mm,
            ac=1400., mini=state.mini_condenser.z_mm, sample=state.sample.upper_surface_z_mm)
        self.names = sorted(self.planes, key=self.planes.get)
        self.landmarks = np.array([self.planes[key] for key in self.names])
        z = np.unique(np.r_[np.linspace(state.electron_gun.exit_plane_z_mm, self.planes["sample"], 18000), self.landmarks])*.001
        self.lengths, self.marks = np.diff(z), np.searchsorted(z, self.landmarks*.001)
        points = np.c_[np.zeros((len(z)-1, 2)), .5*(z[1:]+z[:-1])]
        fields = []
        self.end_field_per_percent = []
        for part in self.lenses:
            field = capture_round_lens_potential(state, part.key).potential.field
            local = field.registration.positions_global_to_local_m(points)
            lo, hi = np.asarray(field.native_support_mm)*.001
            valid = (local[:,2] > lo) & (local[:,2] < hi)
            values = np.zeros(len(points))
            values[valid] = field.field_at_global_positions_t(points[valid])[:,2]
            fields.append(values/part.percent)
            endpoint = np.array([[0., 0., self.planes["sample"]*.001]])
            loc = field.registration.positions_global_to_local_m(endpoint)[0,2]
            self.end_field_per_percent.append(float(field.field_at_global_positions_t(endpoint)[0,2]/part.percent) if lo < loc < hi else 0.)
        self.fields = np.ascontiguousarray(fields)
        self.end_field_per_percent = np.array(self.end_field_per_percent)
        self.rigidity = .5*_charge_over_momentum(state.electron_gun.nominal_exit_energy_ev)
        with np.load(GUN / "source.npz", allow_pickle=False) as data:
            self.xy = np.c_[data["gun_exit_x_m"], data["gun_exit_y_m"]]
            self.slopes = np.c_[data["gun_exit_tx_rad"], data["gun_exit_ty_rad"]]
            self.weights, self.alive = data["gun_exit_weight"], data["gun_exit_alive"]
        xy = self.xy-np.average(self.xy, axis=0, weights=self.weights)
        slopes = self.slopes-np.average(self.slopes, axis=0, weights=self.weights)
        curvature = np.sum(self.weights[:,None]*xy*slopes)/np.sum(self.weights[:,None]*xy**2)
        self.pencil = np.array((1., curvature))

    def measure(self, drives):
        m = matrices(self.fields, drives, self.lengths, self.marks, self.rigidity)
        maps = dict(zip(self.names, m))
        ap = maps["c2ap"]
        pupil = ap[0,0]*self.xy+ap[0,1]*self.slopes
        aperture = self.state.condenser_aperture_2
        keep = self.alive & (np.linalg.norm(pupil-np.array((aperture.offset_x_mm,aperture.offset_y_mm))*.001, axis=1) < aperture.radius_mm*.001)
        last = maps["sample"]
        positions = last[0,0]*self.xy+last[0,1]*self.slopes
        canonical = last[1,0]*self.xy+last[1,1]*self.slopes
        # Rotation magnitude is sufficient for the axisymmetric proposal cone.
        g = self.rigidity*float(self.end_field_per_percent@drives)
        mechanical = canonical-g*np.c_[-positions[:,1], positions[:,0]]
        if keep.sum() < 2:
            return dict(count=int(keep.sum()), d95_um=0., alpha95_mrad=1e6), maps
        stats = transverse_beam_statistics(positions[:,0],positions[:,1],mechanical[:,0],mechanical[:,1],alive=keep,weights=self.weights)
        centered = positions[keep]-np.average(positions[keep],axis=0,weights=self.weights[keep])
        residual = canonical[keep]-np.average(canonical[keep],axis=0,weights=self.weights[keep])
        curvature = np.sum(self.weights[keep,None]*centered*residual)/np.sum(self.weights[keep,None]*centered**2)
        return dict(count=int(keep.sum()), current_fraction=float(self.weights[keep].sum()),
            d95_um=float(stats.radius_95_m*2e6), rms_radius_um=float(stats.radius_rms_m*1e6),
            alpha95_mrad=float(stats.convergence_95_mrad), curvature_per_m=float(curvature)), maps

    def drives(self, keys, values, base=None):
        drives = self.base.copy() if base is None else np.array(base).copy()
        for key, value in zip(keys, values):
            drives[self.keys.index(key)] = value
        return drives


def main():
    proposal = Proposal()
    keys = ("condenser_lens_2", "condenser_lens_3", "adapter_lens", "probe_tl12_lens")
    candidates, attempts = [], []
    seeds = [(9.83,90.,a,14.) for a in (7.,9.2,12.,18.,25.)]
    seeds += [(10.56,33.6,9.2,t) for t in (10.,25.,40.,55.,75.,95.)]
    seeds += [(c2,c3,15.,t) for c2,c3 in ((8.,20.),(10.,55.),(11.,40.)) for t in (20.,50.,80.)]
    print("Prepared signed finite axial fields and full saved source for first-order proposals", flush=True)
    baseline, _ = proposal.measure(proposal.base)
    for seed in seeds:
        def residual(values):
            drives = proposal.drives(keys, values)
            row, maps = proposal.measure(drives)
            hpheight = abs((maps["hp2"]@proposal.pencil)[0])
            central = (maps["central"]@proposal.pencil)[0]/max(hpheight,1e-3)
            return (np.log(max(row["d95_um"],1e-6)), row.get("curvature_per_m",1e6)/1000., central)
        fit = least_squares(residual, seed, bounds=(np.full(4,.1),np.full(4,100.)),
                            diff_step=1e-5, max_nfev=180, xtol=1e-10,ftol=1e-9,gtol=1e-8)
        drives = proposal.drives(keys, fit.x)
        row, maps = proposal.measure(drives)
        attempts.append(dict(seed=seed, drives=fit.x.tolist(), residual=fit.fun.tolist(), measurement=row))
        if (np.linalg.norm(fit.fun) > .03 or row["count"] < 100
                or any(np.linalg.norm(drives-np.array(old["drives"])) < 1e-4 for old in candidates)):
            continue
        item=dict(drives=drives.tolist(), controls={key:dict(percent=float(drives[i])) for i,key in enumerate(proposal.keys)},
            residual=fit.fun.tolist(), measurement=row, planes=proposal.planes,
            pencil_heights_and_slopes={key:(value@proposal.pencil).tolist() for key,value in maps.items()})
        candidates.append(item)
        print(json.dumps(dict(seed=seed, controls={key:item["controls"][key] for key in keys},**row)),flush=True)
    report=dict(scope="First-order proposal only; nonlinear full-source and driven tests still required",
        baseline=baseline, source_original_positive_count=int(np.count_nonzero(proposal.weights>0)),
        keys=proposal.keys, base=proposal.base.tolist(), candidates=candidates,
        best_attempts=sorted(attempts,key=lambda row:np.linalg.norm(row["residual"]))[:12])
    (HERE/"proposals.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print("Proposals:",len(candidates),flush=True)
    print(json.dumps(report["best_attempts"][:4],indent=2),flush=True)


if __name__ == "__main__":
    main()

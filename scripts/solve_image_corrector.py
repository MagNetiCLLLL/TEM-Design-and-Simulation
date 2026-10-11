"""Bounded candidate search for image-corrector conjugacy; never saves defaults.

The scalar constant-energy Larmor equation is a fast candidate generator only.
Candidates require the independent full-field/CUDA validator before adoption.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.optimize import least_squares
from threadpoolctl import threadpool_limits
from validate_image_corrector import save, inventory, ROOT

KEYS=("image_tl11_lens","image_tl12_lens","image_tl21_lens","image_tl22_lens","image_adapter_lens")


class LarmorModel:
    def __init__(self,state,step_mm):
        from temsim.physics.core import fields, interleaved_rk4_values, electron
        from temsim.optics.direct_alignment import _piecewise_endpoint_exact_grid
        from temsim.physics.lens_field_provider import active_vector_providers
        from temsim.physics.posed_column_fields import capture_posed_column_fields
        if active_vector_providers(state) or capture_posed_column_fields(state):
            raise ValueError("Candidate model requires aligned analytic round fields")
        self.state=state
        by_key={p.key:p for p in state.image_corrector_system.components}
        self.lenses=[by_key[key] for key in KEYS]
        self.planes=dict(sample=float(state.sample.z_mm),
            first_middle=(by_key[KEYS[0]].z_mm+by_key[KEYS[1]].z_mm)/2,
            hp1=float(by_key['image_hp1_hexapole'].z_mm),
            middle=(by_key[KEYS[2]].z_mm+by_key[KEYS[3]].z_mm)/2,
            hp2=float(by_key['image_hp2_hexapole'].z_mm),sad=float(by_key['image_sad_plane'].z_mm))
        self.names=sorted((n for n in self.planes if n!='sample'),key=self.planes.get)
        targets=[self.planes[n] for n in self.names]
        self.z_mm=_piecewise_endpoint_exact_grid(self.planes['sample'],self.planes['sad'],step_mm,targets)
        self.z_m=np.ascontiguousarray(self.z_mm*1e-3)
        self.indices=np.asarray(np.searchsorted(self.z_mm,targets),dtype=np.int64)
        stages=interleaved_rk4_values(self.z_mm,.5*(self.z_mm[1:]+self.z_mm[:-1]))
        original=np.array([p.percent for p in self.lenses],float)
        self.original=original
        self.maximum=np.array([p.max_percent for p in self.lenses],float)
        try:
            for lens in self.lenses:lens.percent=0.
            self.fixed,sx,sy=fields(stages,state)
            if np.max(np.abs(sx))+np.max(np.abs(sy))>1e-10:
                raise ValueError("Candidate model requires zero quadrupole focusing")
            basis=[]
            for lens in self.lenses:
                lens.percent=100.
                basis.append(fields(stages,state)[0]-self.fixed)
                lens.percent=0.
            self.basis=np.asarray(basis)
        finally:
            for lens,value in zip(self.lenses,original):lens.percent=float(value)
        charge,momentum,_=electron(state)
        self.factor=charge/(2*momentum)

    def maps(self,percent):
        from temsim.optics.direct_alignment import _rk4_axisymmetric_larmor_matrices
        b=self.fixed+np.asarray(percent)@self.basis/100
        g=np.ascontiguousarray(self.factor*b)
        raw=_rk4_axisymmetric_larmor_matrices(g,self.z_m,self.indices)
        phases=np.r_[0.,np.cumsum((g[:-2:2]+4*g[1::2]+g[2::2])*np.diff(self.z_m)/6)]
        values={name:raw[i] for i,name in enumerate(self.names)}
        values['relay']=values['hp2']@np.linalg.inv(values['hp1'])
        angles={name:float(-phases[self.indices[i]]) for i,name in enumerate(self.names)}
        angles['relay']=angles['hp2']-angles['hp1']
        return values,angles

    def residual(self,percent):
        values,_=self.maps(percent)
        return np.array((values['first_middle'][0,1]/.01, values['middle'][0,1]/.01,
                         values['relay'][0,1]/.01, values['relay'][0,0]+1., values['sad'][0,1]/.01))


def solve(args):
    from temsim.assembly_catalog import AssemblyCatalog,AssemblySelection
    from temsim.optics.column import default_state
    from temsim.instrument_snapshot import capture_instrument_snapshot
    before=inventory()
    state=default_state();AssemblyCatalog().apply(state,AssemblySelection('FEG',args.variant,'No Energy Filter'))
    args.output.mkdir(parents=True,exist_ok=True)
    save(args.output/'inputs.json',capture_instrument_snapshot(state).to_dict())
    start=perf_counter()
    with threadpool_limits(1):
        model=LarmorModel(state,args.step)
        rng=np.random.default_rng(48171)
        seeds=[model.original,np.full(5,40.),np.full(5,80.)]
        seeds.extend(rng.uniform(15.,95.,(max(0,args.starts-len(seeds)),5)))
        rows=[]
        for index,seed in enumerate(seeds[:args.starts]):
            result=least_squares(model.residual,np.minimum(model.maximum-1e-7,np.maximum(seed,1e-7)),
                bounds=(np.full(5,.001),model.maximum),max_nfev=200,ftol=1e-11,xtol=1e-11,gtol=1e-11)
            maps,angles=model.maps(result.x)
            row=dict(start_index=index,success=bool(result.success),cost=float(result.cost),nfev=int(result.nfev),
                     residual=result.fun,percent=dict(zip(KEYS,result.x)),larmor_matrices=maps,
                     larmor_rotation_deg={k:np.degrees(v) for k,v in angles.items()})
            rows.append(row)
            print(index,row['cost'],row['percent'],flush=True)
        rows.sort(key=lambda r:r['cost'])
        fine=LarmorModel(state,args.step/2)
        candidate=np.array([rows[0]['percent'][k] for k in KEYS])
        fine_maps,fine_angles=fine.maps(candidate)
    report=dict(status='CANDIDATES_ONLY_NOT_ADOPTED',variant=args.variant,step_mm=args.step,
        scope='Aligned analytic constant-energy Larmor candidate generator; no full-field, finite-ray, wall, wave or correction qualification. Laboratory rotation is recorded, not fitted away.',
        fixed_conditions=['sample to first TL-pair midpoint B=0','sample to second TL-pair midpoint B=0',
                          'HP1 to HP2 B=0','HP1 to HP2 radial A=-1','sample to SAD B=0'],
        reference_planes_mm=model.planes,variable_keys=KEYS,original_percent=dict(zip(KEYS,model.original)),
        candidates=rows,best_half_step=dict(residual=fine.residual(candidate),larmor_matrices=fine_maps,
                    larmor_rotation_deg={k:np.degrees(v) for k,v in fine_angles.items()}),
        elapsed_s=perf_counter()-start,source_and_saved_inputs_unchanged=before==inventory())
    save(args.output/'candidate-report.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/image_corrector_20261011/candidates')
    parser.add_argument('--variant',default='C3 + Image Corrector')
    parser.add_argument('--step',type=float,default=.05)
    parser.add_argument('--starts',type=int,default=8)
    solve(parser.parse_args())

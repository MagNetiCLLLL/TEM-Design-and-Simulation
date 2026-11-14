"""Independent zero-current Gaussian-lens variational reference sweep.

This helper emits and transports no finite source. It uses the existing
paraxial first-focus and independent first/third-order Lorentz ODE helpers.
Coefficients describe the prescribed ideal field and reference conjugacy,
not an installed microscope, material model, fitted clipped beam or preset.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from temsim.calculation_manifest import solver_source_identity
from temsim.immutable_json import json_digest
from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.lens_field_provider import CoordinateRegistration,FrozenAnalyticField
from temsim.physics.round_lens_aberration_benchmark import (
    _charge_over_momentum,paraxial_focus_reference,third_order_reference,
)

ENERGY_EV=300000.
ENERGY_RELATIVE_STEP=.0005
PEAKS_T=(.15,.30,.60)
SIGMAS_MM=(3.,6.,12.)
GATES=dict(relative_cs=1e-7,relative_cc=1e-6,absolute_focus_over_sigma=1e-9,
           relative_pupil_conversion=1e-8,linear_focus_residual=1e-9)


def write_json(path,data):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def relative(a,b):return abs(a-b)/max(abs(b),np.finfo(float).tiny)


def compare(a,b,sigma_m):
    metrics=dict(relative_cs=relative(a['coefficients']['cs_mm'],b['coefficients']['cs_mm']),
        relative_cc=relative(a['coefficients']['cc_mm'],b['coefficients']['cc_mm']),
        absolute_focus_over_sigma=abs(a['coefficients']['image_z_m']-b['coefficients']['image_z_m'])/sigma_m,
        relative_pupil_conversion=relative(a['pupil']['initial_radius_m_per_pupil_slope_rad'],b['pupil']['initial_radius_m_per_pupil_slope_rad']))
    return dict(metrics=metrics,gates={key:metrics[key]<=GATES[key] for key in metrics},
        passed=all(metrics[key]<=GATES[key] for key in metrics),
        scope='Numerical repeatability of this zero-current ODE reference only; not finite-source or material-field acceptance')


def run_variant(field,peak_t,sigma_m,tail_sigma,step_divisor):
    qp=abs(_charge_over_momentum(ENERGY_EV));kappa=.5*qp*peak_t*sigma_m
    thin_focal_m=1/((.5*qp*peak_t)**2*sigma_m*math.sqrt(math.pi))
    search_end=max(12*sigma_m,4*thin_focal_m+12*sigma_m)
    options=dict(energy_ev=ENERGY_EV,z_start_m=-tail_sigma*sigma_m,
        focus_search_end_m=search_end,maximum_step_m=sigma_m/step_divisor)
    tick=time.perf_counter()
    focus=paraxial_focus_reference(field,**options)
    coefficient=third_order_reference(field,**options,energy_relative_step=ENERGY_RELATIVE_STEP)
    conversion=1/abs(focus['radial_slope_per_m'])
    residual=float(np.linalg.norm(coefficient['linear_focus_residual_per_initial_height']))
    alpha_reference=[]
    for diameter_mm in (1.5,2.,5.76):
        allowed=min(.00095,diameter_mm*.0005)
        slope=allowed/conversion
        alpha_reference.append(dict(tube_diameter_mm=diameter_mm,
            declared_input_disk_radius_m=.00095,paraxial_admitted_input_radius_m=allowed,
            paraxial_edge_pupil_slope_rad=slope,paraxial_edge_mechanical_angle_mrad=1000*math.atan(slope),
            scope='First-order geometric admission only; no source emitted and no current measured'))
    row=dict(request=options,energy_relative_step=ENERGY_RELATIVE_STEP,dimensionless_kappa=kappa,
        input_tail_sigma=tail_sigma,maximum_step_over_sigma=1/step_divisor,
        b0_at_input_over_peak=math.exp(-.5*tail_sigma**2),thin_lens_focal_estimate_m=thin_focal_m,
        coefficients=coefficient,
        pupil=dict(radial_slope_per_m=focus['radial_slope_per_m'],
            initial_radius_m_per_pupil_slope_rad=conversion,
            initial_radius_mm_per_pupil_mrad=conversion,
            outward_pupil_slopes_per_initial_position_m1=(-focus['image_slopes_per_initial_position_m1']).tolist(),
            initial_mechanical_slopes_per_position_m1=focus['initial_slopes_per_position_m1'].tolist(),
            larmor_angle_rad=focus['larmor_angle_rad'],tube_reference=alpha_reference),
        dimensionless_result=dict(cs_over_sigma=coefficient['cs_mm']/(1000*sigma_m),
            cc_over_sigma=coefficient['cc_mm']/(1000*sigma_m),focus_z_over_sigma=coefficient['image_z_m']/sigma_m,
            pupil_slope_times_sigma=focus['radial_slope_per_m']*sigma_m),
        elapsed_s=time.perf_counter()-tick,
        checks=dict(finite_coefficients=bool(np.isfinite([coefficient['cs_mm'],coefficient['cc_mm'],coefficient['image_z_m'],conversion]).all()),
            positive_cs=coefficient['cs_mm']>0.,positive_cc=coefficient['cc_mm']>0.,
            first_focus_inside_search=options['z_start_m']<coefficient['image_z_m']<search_end,
            paraxial_and_third_order_image_plane_exact=focus['image_z_m']==coefficient['image_z_m'],
            bounded_linear_focus_residual=residual<=GATES['linear_focus_residual']),
        linear_focus_residual_per_initial_height_norm=residual,
        field_digest_before=field.digest,field_digest_after=field.digest)
    row['passed_numerical_checks']=all(row['checks'].values())
    return row


def scaling_checks(cases):
    groups={}
    for case in cases:
        # Equal B*sigma values are exactly the same dimensionless ideal lens.
        key=f"{case['peak_t']*case['sigma_mm']:.12g}"
        groups.setdefault(key,[]).append(case)
    result=[]
    for key,group in groups.items():
        if len(group)<2:continue
        reference=group[0]['variants']['refined10']['dimensionless_result']
        rows=[]
        for case in group[1:]:
            values=case['variants']['refined10']['dimensionless_result']
            errors={name:relative(values[name],reference[name]) for name in reference}
            rows.append(dict(case=case['name'],relative_errors=errors,passed=all(value<=1e-7 for value in errors.values())))
        result.append(dict(equal_peak_times_sigma_t_mm=float(key),reference=group[0]['name'],comparisons=rows,
            passed=all(row['passed'] for row in rows)))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'variational-reference')
    parser.add_argument('--case',action='append',default=[],metavar='PEAK_T,SIGMA_MM')
    parser.add_argument('--budget-s',type=float,default=600.)
    args=parser.parse_args(argv)
    if not math.isfinite(args.budget_s) or args.budget_s<=0:parser.error('Positive finite budget required')
    requested=[(peak,sigma) for peak in PEAKS_T for sigma in SIGMAS_MM]
    if args.case:
        requested=[]
        for text in args.case:
            try:peak,sigma=map(float,text.split(','))
            except ValueError:parser.error('--case must be PEAK_T,SIGMA_MM')
            if peak not in PEAKS_T or sigma not in SIGMAS_MM:parser.error('Case lies outside the declared nine-case numerical domain')
            requested.append((peak,sigma))
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();implementation=solver_source_identity()
    report=dict(schema='ideal-gaussian-variational-reference-v1',scope=__doc__,status='RUNNING',
        new_emitted_physical_particles=0,new_finite_source_trajectories=0,new_coherent_propagations=0,
        implementation=implementation,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        driver_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        code_sha256={str((ROOT/'src/temsim/physics'/name).relative_to(ROOT)).replace('\\','/'):sha256((ROOT/'src/temsim/physics'/name).read_bytes()).hexdigest()
            for name in ('round_lens_aberration_benchmark.py','high_order_round_lens.py','lens_field_provider.py')},
        units=dict(length='m unless suffixed _mm',peak_field='T',energy='eV',coefficients='mm',pupil='paraxial slopes in rad'),
        source_conjugacy='Parallel pencil in the Larmor frame at declared negative input plane; first paraxial focus event only',
        sign_convention='Outward pupil; delta_r=-Cs*theta*|theta|^2+Cc*theta*(delta_E/E0)',
        gates=GATES,cases=[])
    write_json(args.output/'report.json',report)
    try:
        for peak,sigma_mm in requested:
            if time.perf_counter()-started>=args.budget_s:raise TimeoutError('Reference sweep reached its explicit budget between cases')
            sigma=sigma_mm*.001;name=f'peak{peak:g}T-sigma{sigma_mm:g}mm'
            radius=min(.5*sigma,.004)
            native=FrozenAnalyticField('ideal-gaussian-round',((peak,0.,sigma),),CoordinateRegistration(),(-7*sigma_mm,7*sigma_mm),radius)
            field=HighOrderRoundLensPotential(native,radial_order=3,valid_radius_m=radius,axial_domain='gaussian_all_z')
            case=dict(name=name,peak_t=peak,sigma_mm=sigma_mm,field=field.to_dict(),field_digest=field.digest,
                field_dict_sha256=json_digest(field.to_dict()),valid_radius_m=radius,variants={})
            report['cases'].append(case)
            for label,tail,divisor in (('coarse8',8.,32),('refined8',8.,64),('refined10',10.,64)):
                if time.perf_counter()-started>=args.budget_s:raise TimeoutError('Reference sweep reached its explicit budget between variants')
                print(f'{time.perf_counter()-started:.2f}s {name} {label}',flush=True)
                case['variants'][label]=run_variant(field,peak,sigma,tail,divisor)
                write_json(args.output/'report.json',report)
            case['step_repeatability']=compare(case['variants']['coarse8'],case['variants']['refined8'],sigma)
            case['input_tail_repeatability']=compare(case['variants']['refined8'],case['variants']['refined10'],sigma)
            case['passed_numerical_reference_checks']=case['step_repeatability']['passed'] and case['input_tail_repeatability']['passed'] and all(row['passed_numerical_checks'] for row in case['variants'].values())
            print(f'{name}: Cs={case["variants"]["refined10"]["coefficients"]["cs_mm"]:.12g}mm checks={case["passed_numerical_reference_checks"]}',flush=True)
            write_json(args.output/'report.json',report)
        if solver_source_identity()!=implementation:raise ValueError('Solver source changed during the zero-current sweep')
        report['dimensionless_scaling_checks']=scaling_checks(report['cases'])
        report['passed_numerical_reference_checks']=all(row['passed_numerical_reference_checks'] for row in report['cases']) and all(row['passed'] for row in report['dimensionless_scaling_checks'])
        report['status']='COMPLETED_ZERO_CURRENT_REFERENCE'
    except BaseException as exc:
        report['status']='INCOMPLETE_REFERENCE';report['error']=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        report['elapsed_s']=time.perf_counter()-started
        report['qualification']='Prescribed ideal field third-order reference only; no finite-source, clipped-beam, field-order, material/OEM or coherent qualification'
        write_json(args.output/'report.json',report)
    print(json.dumps(dict(status=report['status'],cases=len(report['cases']),passed=report['passed_numerical_reference_checks'],elapsed_s=report['elapsed_s']),indent=2))


if __name__=='__main__':main()

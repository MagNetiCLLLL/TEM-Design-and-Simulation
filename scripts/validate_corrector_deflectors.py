"""Bounded CUDA validation of the five static probe-corrector dipoles.

The default command only prepares a protocol. --execute explicitly runs one
physical Tip/gun source, one shared pre-corrector prefix, baseline relay/focus
checks, and short same-population finite-coil perturbations. No optical or source
default is fitted or saved. Independent DOP853 reference integration covers a
declared subset of the same entrance rays and the complete local scalar fields.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import tomllib
from pathlib import Path
from time import perf_counter

import numpy as np

from validate_probe_relay import (
    LIMITS as RELAY_LIMITS, assert_cuda, save_report, source_identity, subset,
    surface_measurement,
)


COMPONENT_KEYS = (
    "probe_dph2_deflector", "probe_dp22_deflector", "probe_dp21_deflector",
    "probe_dph1_deflector", "probe_dp11_deflector",
)
LIMITS = dict(relative_linear_response=1e-3, relative_even_response=1e-3,
              relative_independent_response=1e-3, reference_refinement_change=1e-5,
              minimum_response_norm=1e-3, minimum_common_current_fraction=.999,
              disabled_response_exact=True)


def topology_intervals(assembly):
    """Physical clear gaps, declared before examining the executed ray fan."""
    result=[]
    for name,left,right in (
        ("before_ADL","condenser_lens_3","adapter_lens"),
        ("between_TL22_TL21","probe_tl22_lens","probe_tl21_lens"),
        ("between_TL1_CM","probe_tl12_lens","mini_condenser"),
    ):
        lower=float(assembly.part(left).end_z_mm)
        upper=float(assembly.part(right).start_z_mm)
        if not lower < upper:
            raise ValueError(f"No clear physical gap for {name}: {lower:g} to {upper:g} mm")
        result.append(dict(name=name,left_part=left,right_part=right,interval_mm=[lower,upper]))
    return result


def manifest_comparison(directory):
    """Record the two original manifests without pretending they were executed."""
    if directory is None:return dict(status="NOT_SUPPLIED")
    base=Path(__file__).resolve().parents[1]/"configs"/"instruments"/"column"
    rows=[]
    for before in sorted(directory.glob("before_*.toml")):
        after=base/before.name.removeprefix("before_")
        old_bytes,new_bytes=before.read_bytes(),after.read_bytes()
        old,new=tomllib.loads(old_bytes.decode()),tomllib.loads(new_bytes.decode())
        old_parts={p["key"]:p for p in old["parts"]}
        new_parts={p["key"]:p for p in new["parts"]}
        changes=[]
        for key in sorted(old_parts.keys()|new_parts.keys()):
            old_part,new_part=old_parts.get(key,{}),new_parts.get(key,{})
            fields=[dict(field=f,before=old_part.get(f),after=new_part.get(f))
                for f in sorted(old_part.keys()|new_part.keys()) if old_part.get(f)!=new_part.get(f)]
            if fields:changes.append(dict(key=key,changes=fields))
        allowed_channels={"physical_host_key","physical_host_status","mechanical_outer_diameter_mm",
                          "axial_relation_note","radial_relation_note"}
        allowed_position={"optical_reference_local_z_mm","local_center_z_mm","local_end_z_mm","local_start_z_mm"}
        def expected(row):
            key=row["key"];changed={f["field"] for f in row["changes"]}
            if key in COMPONENT_KEYS:
                return changed<=allowed_channels|(allowed_position if "dph" in key else set())
            if key in ("probe_hp1_hexapole","probe_hp2_hexapole"):
                return changed=={"mechanical_inner_diameter_mm"}
            return key in {k.replace("_deflector","_coils") for k in COMPONENT_KEYS} and key not in old_parts
        # Reference notes added during review do not change executable geometry.
        geometry_notes={"probe_corrector_coupled_optics_extent_keys","probe_corrector_core_extent_keys",
            "probe_corrector_with_upstream_alignment_extent_keys","probe_corrector_length_comparison_note"}
        old_geometry={k:v for k,v in old.get("geometry",{}).items() if k not in geometry_notes and not k.startswith("probe_corrector_reference_")}
        new_geometry={k:v for k,v in new.get("geometry",{}).items() if k not in geometry_notes and not k.startswith("probe_corrector_reference_")}
        unchanged_other=old_geometry==new_geometry and all(old[k]==new.get(k) for k in old if k not in ("parts","geometry"))
        same_optics=unchanged_other and all(expected(row) for row in changes)
        if not same_optics:raise ValueError(f"Undeclared physical difference in {after.name}")
        rows.append(dict(before=str(before.resolve()),after=str(after.resolve()),
            before_sha256=hashlib.sha256(old_bytes).hexdigest(),after_sha256=hashlib.sha256(new_bytes).hexdigest(),
            unchanged_optical_parameters=True,changes=changes))
    if not rows:raise ValueError("No before_*.toml manifests found")
    return dict(status="DICTIONARY_COMPARISON_PASSED",manifests=rows,
        scope="Only saved column manifests compared; no old-source execution. Lens strengths, HP strengths/centres and every other pre-existing field unchanged except the listed authorised DPH placements, host/body metadata and HP material inner diameter.")


def phase_points(checkpoints, z):
    index = np.flatnonzero(np.isclose(checkpoints.z_mm, z, rtol=0., atol=1e-10))
    if len(index) != 1:
        raise ValueError(f"Missing unique exact checkpoint at {z:g} mm")
    i = int(index[0])
    return i, np.stack([getattr(checkpoints, name)[i] for name in ("x_m", "y_m", "tx_rad", "ty_rad")])


def relative_response_error(actual, reference, weights, length_m):
    """Weighted norm in (x/L,y/L,theta_x,theta_y), never mixed SI units."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    weights = np.asarray(weights, float)
    if actual.shape != reference.shape or actual.shape[0] != 4 or actual.shape[-1] != weights.size:
        raise ValueError("Response arrays must be matching (4, ..., rays)")
    if not math.isfinite(length_m) or length_m <= 0 or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Response comparison needs a positive length and positive current")
    scale = np.array((1/length_m, 1/length_m, 1., 1.)).reshape((4,)+(1,)*(actual.ndim-1))
    a, b = actual*scale, reference*scale
    axes = tuple(range(actual.ndim-1))
    w = weights/weights.sum()
    denominator = float(w @ np.sum(b*b, axis=axes))
    error = float(w @ np.sum((a-b)**2, axis=axes))
    cosine_denominator = math.sqrt(float(w @ np.sum(a*a, axis=axes))*denominator)
    return dict(relative_error=math.sqrt(error/denominator) if denominator else None,
                reference_response_norm=math.sqrt(denominator),
                polarity_cosine=float(w @ np.sum(a*b, axis=axes))/cosine_denominator if cosine_denominator else None)


def independent_reference(initial, momentum, reference_momentum, interval_m,
                          coefficients, drive_support_m, *, breaks_m=(),
                          cs_events=(), rtol=2e-10, command_rad=(0., 0.)):
    """Integrate a continuous canonical ODE and its two drive tangents.

    This does not call any production particle integrator/stepper. ``coefficients``
    supplies (Bz,kx,ky,kxy,hn,hs,Bx,By) at an SI position and an open-interval
    midpoint; finite dipole supports use that midpoint at discontinuous edges.
    k/hex coefficients are at reference momentum. The target command is the
    physical integrated reference-energy angle in radians. Cs events retain
    their original position and coefficient and act once after continuous flow.
    """
    from scipy.constants import elementary_charge
    from scipy.integrate import solve_ivp
    q = -elementary_charge
    initial = np.asarray(initial, float)
    p = np.asarray(momentum, float)
    if initial.shape != (4, p.size) or np.any(p <= 0) or not np.isfinite(initial).all():
        raise ValueError("Independent reference needs finite same-ray coordinates and momenta")
    lower, upper = map(float, interval_m)
    drive_lower, drive_upper = map(float, drive_support_m)
    if not lower < upper or not drive_lower < drive_upper:
        raise ValueError("Independent reference intervals must have positive length")
    ratio = reference_momentum/p
    edges = sorted({lower,upper,*[float(z) for z in breaks_m if lower < z < upper],
                    *[float(z) for z in drive_support_m if lower < z < upper],
                    *[float(z) for z,_ in cs_events if lower < z <= upper]})
    x,y,tx,ty = initial
    g0 = q*float(coefficients(lower,(edges[0]+edges[1])/2)[0])/(2*p)
    value = np.zeros((12,p.size))
    value[:4] = (x,y,tx-g0*y,ty+g0*x)
    evaluations = 0
    atol = np.repeat(np.array((1e-16,1e-16,1e-13,1e-13,
                              1e-12,1e-12,1e-12,1e-12,1e-9,1e-9,1e-9,1e-9)),p.size)
    # Tangents are arranged (canonical component, X/Y command, ray).
    for start,stop in zip(edges[:-1],edges[1:]):
        middle = (start+stop)/2
        forcing = ratio/(drive_upper-drive_lower) if drive_lower < middle < drive_upper else np.zeros_like(p)
        def rhs(z, flat):
            a = flat.reshape(12,p.size)
            x,y,px,py = a[:4]
            tangent = a[4:].reshape(4,2,p.size)
            bz,kx,ky,kxy,hn,hs,bx,by = coefficients(z,middle)
            g = q*bz/(2*p)
            kx,ky,kxy,hn,hs = (float(v)*ratio for v in (kx,ky,kxy,hn,hs))
            fx,fy = -q*by/p+forcing*command_rad[0],q*bx/p+forcing*command_rad[1]
            result = np.empty_like(a)
            result[:4] = (px+g*y,py-g*x,
                -(kx+g*g)*x-kxy*y+g*py-hn*(x*x-y*y)-2*hs*x*y+fx,
                -(ky+g*g)*y-kxy*x-g*px+2*hn*x*y-hs*(x*x-y*y)+fy)
            jacobian = np.zeros((4,4,p.size))
            jacobian[0,1],jacobian[0,2] = g,1.
            jacobian[1,0],jacobian[1,3] = -g,1.
            jacobian[2,0] = -(kx+g*g)-2*hn*x-2*hs*y
            jacobian[2,1] = -kxy+2*hn*y-2*hs*x
            jacobian[2,3] = g
            jacobian[3,0] = -kxy+2*hn*y-2*hs*x
            jacobian[3,1] = -(ky+g*g)+2*hn*x+2*hs*y
            jacobian[3,2] = -g
            derivative = np.einsum("ijn,jkn->ikn",jacobian,tangent)
            derivative[2,0] += forcing
            derivative[3,1] += forcing
            result[4:] = derivative.reshape(8,p.size)
            return result.ravel()
        solution = solve_ivp(rhs,(start,stop),value.ravel(),method="DOP853",rtol=rtol,atol=atol,
                             max_step=(stop-start)/4)
        if not solution.success:
            raise ValueError("Independent DOP853 reference failed: "+solution.message)
        evaluations += solution.nfev
        value = solution.y[:,-1].reshape(12,p.size)
        for event_z,cs in cs_events:
            if float(event_z) != stop:
                continue
            x,y = value[:2]
            tangent = value[4:].reshape(4,2,p.size)
            value[2] -= cs*(x*x+y*y)*x
            value[3] -= cs*(x*x+y*y)*y
            tangent[2] -= cs*((3*x*x+y*y)*tangent[0]+2*x*y*tangent[1])
            tangent[3] -= cs*(2*x*y*tangent[0]+(x*x+3*y*y)*tangent[1])
    g1 = q*float(coefficients(upper,(edges[-2]+edges[-1])/2)[0])/(2*p)
    x,y,px,py = value[:4]
    tangent = value[4:].reshape(4,2,p.size).copy()
    tangent[2] += g1*tangent[1]
    tangent[3] -= g1*tangent[0]
    return np.stack((x,y,px+g1*y,py-g1*x)),tangent,dict(
        method="SciPy DOP853 continuous canonical ODE plus analytic two-drive variational equations",
        rtol=rtol,nfev=evaluations,interval_m=[lower,upper],ray_count=p.size,
        production_stepper_used=False,cs_events=[list(row) for row in cs_events])


def independent_electric_reference(initial,energies,reference_momentum,interval_m,
        coefficients,drive_support_m,electric,*,breaks_m=(),cs_events=(),rtol=2e-10,
        command_step_rad=1e-7):
    """Independent dimensional canonical flow including the actual scalar field.

    The physical transverse canonical momentum is divided by each ray's fixed
    entrance momentum only to condition the solver. Local momentum/speed are
    recomputed from K0 + phi(x,y,z) - phi0. The response uses a declared central
    difference independent of the production drive size, checked again at half
    the step and tighter ODE tolerance by the caller.
    """
    from scipy.constants import elementary_charge,electron_mass,speed_of_light
    from scipy.integrate import solve_ivp
    q=-elementary_charge;rest=electron_mass*speed_of_light**2/elementary_charge
    initial=np.asarray(initial,float);energies=np.asarray(energies,float)
    if initial.shape!=(4,len(energies)) or np.any(energies<=0):
        raise ValueError("Electric reference needs actual positive ray energies")
    def momentum_speed(energy):
        a=energy/rest
        return (electron_mass*speed_of_light*np.sqrt(a*(a+2)),
                speed_of_light*np.sqrt(a*(a+2))/(a+1))
    p0,_=momentum_speed(energies);ratio=reference_momentum/p0
    start,end=map(float,interval_m);lo,hi=map(float,drive_support_m)
    edges=sorted({start,end,*[float(z) for z in breaks_m if start<z<end],
        *[float(z) for z in (lo,hi) if start<z<end],*[float(z) for z,_ in cs_events if start<z<=end]})
    maximum_field=0.;maximum_energy_change=0.;maximum_radius=0.;evaluations=0
    def sample(x,y,z):
        nonlocal maximum_field,maximum_radius
        phi,ef=electric.interpolate(np.column_stack((x,y,np.full(len(x),z))))
        maximum_field=max(maximum_field,float(np.max(np.linalg.norm(ef,axis=1))))
        maximum_radius=max(maximum_radius,float(np.max(np.hypot(x,y))))
        return phi,ef
    x,y,tx,ty=initial;phi0,_=sample(x,y,start)
    invariant=energies-phi0
    def pv(x,y,z):
        nonlocal maximum_energy_change
        phi,ef=sample(x,y,z);energy=invariant+phi
        if np.any(energy<=0):raise ValueError("Electric reference left the positive-energy domain")
        maximum_energy_change=max(maximum_energy_change,float(np.max(abs(energy-energies))))
        p,v=momentum_speed(energy)
        return p/p0,v,ef
    def integrate(command):
        nonlocal evaluations
        g0=q*float(coefficients(start,(edges[0]+edges[1])/2)[0])/(2*p0)
        value=np.stack((x,y,tx-g0*y,ty+g0*x))
        for a,b in zip(edges[:-1],edges[1:]):
            midpoint=(a+b)/2
            forcing=ratio/(hi-lo) if lo<midpoint<hi else np.zeros_like(p0)
            def rhs(z,flat):
                xx,yy,px,py=flat.reshape(4,len(p0))
                bz,kx,ky,kxy,hn,hs,bx,by=coefficients(z,midpoint)
                g=q*bz/(2*p0);scale,speed,ef=pv(xx,yy,z)
                fx=-q*by/p0+q*ef[:,0]/(p0*speed)+forcing*command[0]
                fy=q*bx/p0+q*ef[:,1]/(p0*speed)+forcing*command[1]
                return np.array(((px+g*yy)/scale,(py-g*xx)/scale,
                    -ratio*kx*xx-g*g/scale*xx-ratio*kxy*yy+g/scale*py-ratio*(hn*(xx*xx-yy*yy)+2*hs*xx*yy)+fx,
                    -ratio*ky*yy-g*g/scale*yy-ratio*kxy*xx-g/scale*px+ratio*(2*hn*xx*yy-hs*(xx*xx-yy*yy))+fy)).ravel()
            result=solve_ivp(rhs,(a,b),value.ravel(),method="DOP853",rtol=rtol,
                atol=np.repeat((1e-16,1e-16,1e-13,1e-13),len(p0)),max_step=(b-a)/4)
            if not result.success:raise ValueError("Independent electric ODE failed: "+result.message)
            evaluations+=result.nfev;value=result.y[:,-1].reshape(4,len(p0))
            for z,cs in cs_events:
                if float(z)==b:
                    scale,_,_=pv(value[0],value[1],b)
                    action=cs*(value[0]**2+value[1]**2)*scale
                    value[2]-=action*value[0];value[3]-=action*value[1]
        xx,yy,px,py=value;scale,_,_=pv(xx,yy,end)
        g=q*float(coefficients(end,(edges[-2]+edges[-1])/2)[0])/(2*p0)
        return np.stack((xx,yy,(px+g*yy)/scale,(py-g*xx)/scale))
    centre=integrate((0.,0.));responses=[]
    for axis in (0,1):
        command=np.zeros(2);command[axis]=command_step_rad
        responses.append((integrate(command)-integrate(-command))/(2*command_step_rad))
    return centre,np.stack(responses,axis=1),dict(
        method="Independent SciPy DOP853 dimensional canonical ODE including actual phi and E; central-drive response",
        rtol=rtol,command_step_rad=command_step_rad,nfev=evaluations,ray_count=len(p0),
        electric_provider=type(electric).__name__,maximum_sampled_electric_v_per_m=maximum_field,
        maximum_sampled_energy_change_ev=maximum_energy_change,maximum_sampled_radius_m=maximum_radius,
        production_stepper_used=False,cs_events=[list(row) for row in cs_events])


def components(state):
    by_key = {component.key:component for component in state.corrector_elements}
    result = []
    for key in COMPONENT_KEYS:
        component = by_key.get(key)
        if component is None or not all(hasattr(component,n) for n in ("kick_x_mrad","kick_y_mrad","enabled")):
            raise ValueError(f"Missing static corrector dipole control: {key}")
        result.append(component)
    return result


def prepare(args):
    from temsim.instrument_snapshot import InstrumentSnapshot,capture_instrument_snapshot
    if args.inputs:
        state = InstrumentSnapshot.from_dict(json.loads(args.inputs.read_text(encoding="utf-8"))).restore()
    else:
        from temsim.optics.column import default_state
        from temsim.assembly_catalog import AssemblyCatalog
        from temsim.operating_modes import apply_operating_mode_pair
        state = default_state()
        catalog = AssemblyCatalog();catalog.apply(state,catalog.default_selection())
        apply_operating_mode_pair(state,"nano_probe","diffraction")
    original = dict(simulation_mode=state.simulation_mode,
                    backend=state.acceleration_backend,acceleration_enabled=state.acceleration_enabled,
                    scan_enabled=state.ac_deflector.scan_enabled,descan_enabled=state.descan_deflector.scan_enabled,
                    wobble_enabled=state.ac_deflector.wobble_enabled)
    state.simulation_mode="custom"
    state.ac_deflector.scan_enabled=False;state.descan_deflector.scan_enabled=False
    state.ac_deflector.wobble_enabled=False;state.descan_deflector.wobble_enabled=False
    state.acceleration_enabled=True;state.acceleration_backend="Require GPU"
    state.electron_gun.emitter.ray_count=args.rays
    state.step_mm=args.step
    from temsim.physics.instrument_magnetic import column_dipole_fields
    coils = column_dipole_fields(state)
    if any(coil.dynamic for coil in coils):
        raise ValueError("This static validation cannot silently freeze another dynamic drive")
    rows=[]
    for item in components(state):
        if not item.enabled:
            raise ValueError(f"{item.key}: baseline component must already be enabled")
        if max(abs(item.kick_x_mrad),abs(item.kick_y_mrad))+args.delta_mrad>item.maximum_kick_mrad:
            raise ValueError(f"{item.key}: perturbation exceeds original component limit")
        matches=[c for c in coils if item.key in c.drive_keys]
        if len(matches)!=1 or matches[0].registration is not None:
            raise ValueError(f"{item.key}: independent reference currently requires one native-axis finite coil")
        coil=matches[0];low,high=coil.field_support_mm
        rows.append(dict(key=item.key,name=item.name,baseline_x_mrad=item.kick_x_mrad,
            baseline_y_mrad=item.kick_y_mrad,field_key=coil.key,field_support_mm=[low,high],
            local_interval_mm=[low-.001,high+.001],effective_length_mm=high-low,
            mechanical_length_mm=item.mechanical_length_mm,
            mechanical_clear_bore_diameter_mm=item.mechanical_clear_bore_diameter_mm))
    from temsim.specimen.scene import SpecimenScene
    if not SpecimenScene.from_state(state).is_vacuum:
        raise ValueError("Baseline sample-focus diagnostics require an unchanged vacuum reference sample")
    if getattr(getattr(state,"nanopulser",None),"installed",False):
        raise ValueError("Pulsed source hardware is outside this static validation")
    snapshot=capture_instrument_snapshot(state)
    assembly=state._resolved_assembly
    intervals=topology_intervals(assembly)
    protocol=dict(schema="corrector-static-dipoles-validation-v1",input_identity=snapshot.digest,
        prepared_utc=datetime.now(timezone.utc).isoformat(),components=rows,rays=args.rays,
        perturbation_mrad=args.delta_mrad,maximum_step_mm=args.step,reference_rays=args.reference_rays,
        prefix_z_mm=min(*[r["local_interval_mm"][0] for r in rows],
                        *[r["interval_mm"][0] for r in intervals]),
        crossover_intervals=intervals,crossover_coarse_spacing_mm=1.,
        crossover_local_half_width_mm=.001,
        sample_entrance_mm=float(state.sample.upper_surface_z_mm),
        original_runtime_flags=original,
        detached_runtime_changes="Custom nonlinear model; static AC/Descan (coils and biases retained); mandatory CUDA; declared ray count and numerical step. No source/lens/deflector default fitted or written.",
        limits=LIMITS,baseline_limits=RELAY_LIMITS,
        scope="Same physical source; full-population CUDA local perturbations; independent local reference on declared same-ray subset. Not wave or experimental qualification.")
    return state,snapshot,protocol


def trace(state,start,stop,coordinates,energy,times,alive,stops,keys,planes,step,receipts,label):
    from temsim.physics.core import propagate
    from temsim.physics.backend_execution import backend_receipts
    from temsim.physics.instrument_magnetic import active_column_events
    from validate_ac_wave_rays import _survival
    x,y,tx,ty=coordinates
    rows=[];started=perf_counter()
    with backend_receipts(rows):
        output=propagate(state,start,stop,x,tx,y,ty,events=active_column_events(state),
            include_spherical_aberration=True,include_hexapole=True,maximum_step_mm=step,
            checkpoint_z_mm=planes,save_z_mm=planes,return_checkpoints=True,
            initial_kinetic_energy_ev=energy,initial_time_s=times,return_flight_times=True,
            include_initial_plane_kicks=False,defer_nonfinite_until_clipping=True)
    assert_cuda(rows)
    receipts.extend(dict(row,case=label,elapsed_s=perf_counter()-started) for row in rows)
    end_alive,end_stops,end_keys,masks=_survival(state,output,alive,stops,keys)
    masks &= np.asarray(alive,bool)[None,:]
    return output[-1],masks,end_alive,end_stops,end_keys


def reference_for_component(state,row,coordinates,energies,step,rtol):
    from scipy.constants import elementary_charge,electron_mass,speed_of_light
    from temsim.physics.core import build_propagation_plan,fields,skew_quadrupole_field,hexapole_field_components
    from temsim.physics.instrument_magnetic import active_column_events,column_dipole_fields
    from temsim.physics.electrostatic_column_transport import active_electric_field
    start,stop=row["local_interval_mm"]
    plan=build_propagation_plan(state,start,stop,active_column_events(state),maximum_step_mm=step)
    if plan.mapped_fields or plan.posed_spherical_kicks:
        raise ValueError("Independent scalar reference cannot omit a placed field")
    if any(np.any(getattr(plan,name)) for name in ("thin_power_m1","thin_rotation_rad","kick_x_rad","kick_y_rad")):
        raise ValueError("Local reference expects finite dipoles and explicit Cs, not an extra thin impulse")
    coils=column_dipole_fields(state)
    if any(c.registration is not None for c in coils if c.field_support_mm[0]<stop and c.field_support_mm[1]>start):
        raise ValueError("Independent reference does not silently flatten a posed coil")
    boundaries=[]
    for c in coils:
        boundaries.extend((c.lower_m,c.upper_m))
    for item in (*state.lenses,*state.stigmators,*state.corrector_elements):
        support=getattr(item,"field_support_mm",None)
        if callable(support):
            boundaries.extend(float(z)*1e-3 for z in support())
    def coefficients(z,middle):
        zz=np.array((z*1000.,))
        bz,kx,ky=fields(zz,state)
        kxy=skew_quadrupole_field(zz,state)
        hn,hs=hexapole_field_components(zz,state)
        active=[c for c in coils if c.lower_m<middle<c.upper_m]
        return (float(bz[0]),float(kx[0]),float(ky[0]),float(kxy[0]),float(hn[0]),float(hs[0]),
                sum(c.bx_t for c in active),sum(c.by_t for c in active))
    kinetic=np.asarray(energies)*elementary_charge
    momentum=np.sqrt(kinetic*(kinetic+2*electron_mass*speed_of_light**2))/speed_of_light
    cs=[(float(z)*1e-3,float(c)) for z,c in zip(plan.z_mm,plan.cs_kick_m3) if c and z>start]
    electric=active_electric_field(plan)
    if electric is not None:
        return independent_electric_reference(coordinates,energies,float(plan.reference_momentum_kg_m_s),
            (start*1e-3,stop*1e-3),coefficients,np.asarray(row["field_support_mm"])*1e-3,electric,
            breaks_m=boundaries,cs_events=cs,rtol=rtol,command_step_rad=1e-7 if rtol>=2e-10 else 5e-8)
    return independent_reference(coordinates,momentum,float(plan.reference_momentum_kg_m_s),
        (start*1e-3,stop*1e-3),coefficients,np.asarray(row["field_support_mm"])*1e-3,
        breaks_m=boundaries,cs_events=cs,rtol=rtol)


def select_reference_rays(coordinates,energies,weights,alive,count):
    live=np.flatnonzero(alive & (weights>0))
    if len(live)<5:raise ValueError("Insufficient common live rays for independent reference")
    r2=np.sum(coordinates[:2,live]**2,axis=0)
    order=live[np.argsort(r2)]
    selected=set(order[np.linspace(0,len(order)-1,min(count,len(order))).round().astype(int)].tolist())
    selected.update((int(live[np.argmin(energies[live])]),int(live[np.argmax(energies[live])])))
    return np.asarray(sorted(selected),dtype=int)


def resumed_initial(points,masks,z,stops,keys):
    index,coordinates=phase_points(points,z)
    alive=masks[index].copy()
    # A later stop from a baseline trace is not an initial condition for a
    # perturbed continuation. Earlier stops remain represented by alive=False.
    preceding_stops=np.asarray(stops,float).copy()
    preceding_stops[alive]=np.nan
    preceding_keys=list(keys)
    for i in np.flatnonzero(alive):preceding_keys[i]=""
    return dict(coordinates=coordinates,energy=points.kinetic_energy_ev[index],
        times=points.flight_time_s[index],alive=alive,stops=preceding_stops,keys=preceding_keys)


def refine_local_crossover(state,candidate,coarse,weights,step,receipts,label,half_width):
    """Reuse the executed left boundary; refine only its short actual bracket."""
    from scipy.optimize import brentq
    from temsim.physics.beam_statistics import transverse_beam_statistics
    from temsim.physics.corrector_crossovers import measured_corrector_crossovers
    cp,masks,_,stops,keys=coarse
    left,right=map(float,candidate["bracket_mm"])
    li,_=phase_points(cp,left);ri,_=phase_points(cp,right)
    live=masks[li] & masks[ri] & (weights>0)
    initial=resumed_initial(cp,masks,left,stops,keys)
    calls={}
    def covariance(z):
        if z==left:
            point=cp;index=li;surviving=masks[li]
        else:
            local=trace(state,left,z,**initial,planes=(z,),step=step,receipts=receipts,label=label+":root")
            point=local[0];index,_=phase_points(point,z);surviving=local[1][index]
        if np.any(live & ~surviving):
            raise ValueError("Crossover refinement changed the shared physical population")
        values=[getattr(point,n)[index] for n in ("x_m","y_m","tx_rad","ty_rad")]
        stats=transverse_beam_statistics(*values,weights=weights,alive=live)
        value=float(stats.radial_position_angle_covariance_m_rad)
        calls[float(z)]=value
        return value
    left_cov,right_cov=covariance(left),covariance(right)
    if not left_cov<0<right_cov:
        return dict(status="REFINED_BRACKET_NOT_SIGN_CHANGING",candidate=candidate,
                    endpoint_covariance_m_rad=[left_cov,right_cov],crossovers=[])
    root=float(brentq(covariance,left,right,xtol=1e-8,rtol=1e-13,maxiter=16))
    width=min(half_width,.45*(root-left),.45*(right-root))
    if width<=0:raise ValueError("Refined crossover lies on a bracket boundary")
    planes=(root-width,root,root+width)
    local=trace(state,left,planes[-1],**initial,planes=planes,step=step,
                receipts=receipts,label=label+":three-plane")
    waists=measured_corrector_crossovers(local[0],local[1],weights,left,right)
    point,mask=local[:2]
    common=np.all(mask,axis=0)&(weights>0);w=weights[common]/weights[common].sum()
    xy=np.stack((point.x_m[:,common],point.y_m[:,common]),axis=1)
    xy-=np.sum(xy*w[None,None,:],axis=2,keepdims=True)
    eigenvalues,eigenvectors=np.linalg.eigh((xy[1]*w)@xy[1].T)
    projected=np.einsum("ij,pjn->pin",eigenvectors.T,xy)
    variances=np.sum(projected**2*w[None,None,:],axis=2)
    half_width_m=width*1e-3
    derivative=(variances[2]-variances[0])/(2*half_width_m)
    second=(variances[2]-2*variances[1]+variances[0])/half_width_m**2
    offsets=np.divide(-derivative,second,out=np.full(2,np.nan),where=second>0)
    return dict(status="EXECUTED_LOCAL_REFINEMENT",candidate=candidate,z_mm=root,
        maximum_step_mm=step,physical_planes_mm=list(planes),crossovers=waists,
        principal_axes_at_radial_minimum=dict(rms_radius_nm=(np.sqrt(eigenvalues)*1e9).tolist(),
            basis_columns_lab_xy=eigenvectors.tolist(),bracket_fixed_axis_rms_nm=(np.sqrt(variances)*1e9).tolist(),
            local_quadratic_axis_waist_offsets_mm=(offsets*1e3).tolist(),
            local_quadratic_axis_waist_separation_mm=float(abs(offsets[1]-offsets[0])*1e3),
            interpretation="Three executed planes in a fixed principal basis. Axis waist offsets are local quadratic estimates, not independently executed conjugate planes."),
        covariance_evaluations=[dict(z_mm=z,covariance_m_rad=v) for z,v in sorted(calls.items())],
        scope="Exact local continuation from the same executed coarse left boundary; upstream not retraced")


def baseline(state,protocol,gun,receipts):
    from temsim.optics.direct_alignment import canonical_transfers
    from temsim.optics.beam_path_audit import crossover_candidates
    bundle=gun.exit_bundle;weights=np.asarray(bundle.weight)
    coordinates=np.stack((bundle.x_m,bundle.y_m,bundle.tx_rad,bundle.ty_rad))
    energy=state.beam_voltage_kv*1000.+np.asarray(bundle.energy_offset_ev)
    prefix=protocol["prefix_z_mm"];sample=protocol["sample_entrance_mm"]
    step=protocol["maximum_step_mm"]
    pre=trace(state,float(state.electron_gun.exit_plane_z_mm),prefix,coordinates,energy,
        bundle.flight_time_s,bundle.alive,gun.blocked_z_mm,gun.blocked_key,(prefix,),step,receipts,"common-prefix")
    points,masks,alive,stops,keys=pre
    initial=resumed_initial(points,masks,prefix,stops,keys)
    sample_planes=np.array((sample-.0002,sample-.0001,sample))
    local_planes=[z for row in protocol["components"] for z in row["local_interval_mm"]]
    interval_planes=[z for row in protocol["crossover_intervals"] for z in row["interval_mm"]]
    planes=np.unique(np.r_[local_planes,interval_planes,
        np.arange(prefix,sample,protocol["crossover_coarse_spacing_mm"]),sample_planes])
    coarse=trace(state,prefix,sample,**initial,planes=planes,step=step,receipts=receipts,label="baseline-coarse")
    cp,mask,*_=coarse
    topology=[]
    for row in protocol["crossover_intervals"]:
        lower,upper=row["interval_mm"]
        selected=(cp.z_mm>=lower)&(cp.z_mm<=upper)
        proposals=crossover_candidates(subset(cp,selected),mask[selected],weights)
        refinements=[refine_local_crossover(state,candidate,coarse,weights,step/2,receipts,
            row["name"],protocol["crossover_local_half_width_mm"]) for candidate in proposals]
        topology.append(dict(**row,candidates=proposals,refinements=refinements,
            measured_two_dimensional_waist=any(r["crossovers"] for r in refinements)))
    fine=trace(state,prefix,sample,**initial,planes=sample_planes,step=step/2,receipts=receipts,label="baseline-fine-downstream")
    fcp,fmask,*_=fine
    coarse_focus=surface_measurement(cp,mask,weights,sample_planes)
    fine_focus=surface_measurement(fcp,fmask,weights,sample_planes)
    hp2,hp1=state.hp2_hexapole,state.hp1_hexapole
    transfer=canonical_transfers(state,hp2.z_mm,[hp1.z_mm])[float(hp1.z_mm)]
    magnification=np.linalg.svd(transfer.j_img,compute_uv=False)
    b=float(np.linalg.norm(transfer.j_diff_m_per_rad,ord=2))
    checks=dict(hp_conjugacy=b<=RELAY_LIMITS["hp_b_norm_m_per_rad"] and np.max(abs(magnification-1))<=RELAY_LIMITS["hp_magnification_error"],
        sample_rms=fine_focus["rms_radius_nm"]<RELAY_LIMITS["sample_rms_nm"],
        sample_diameter95=fine_focus["diameter95_nm"]<RELAY_LIMITS["sample_diameter95_nm"],
        sample_covariance_focus=abs(fine_focus["covariance_waist_offset_nm"])<RELAY_LIMITS["sample_waist_offset_nm"],
        sample_derivative_focus=abs(fine_focus["derivative_waist_offset_nm"])<RELAY_LIMITS["sample_waist_offset_nm"] and fine_focus["variance_second_derivative"]>0,
        sample_alpha95=abs(fine_focus["statistics"]["convergence_95_rad"]*1e3/RELAY_LIMITS["sample_alpha95_mrad"]-1)<=RELAY_LIMITS["sample_alpha95_relative_error"],
        step_radius=abs(coarse_focus["rms_radius_nm"]-fine_focus["rms_radius_nm"])<RELAY_LIMITS["step_radius_difference_nm"],
        step_focus=abs(coarse_focus["covariance_waist_offset_nm"]-fine_focus["covariance_waist_offset_nm"])<RELAY_LIMITS["step_waist_difference_nm"])
    checks.update({"waist_"+row["name"]:row["measured_two_dimensional_waist"] for row in topology})
    evidence=dict(checks=checks,hp_b_norm_m_per_rad=b,hp_magnifications=magnification.tolist(),
        source_crossovers=topology,coarse_sample=coarse_focus,fine_sample=fine_focus,
        source_prefix_scope="One actual gun source and declared-step prefix reused. Half-step focus comparison refines the downstream corrector/sample region; each crossover separately refines only its actual saved bracket.")
    return coarse,initial,evidence


def render_figure(output,state,protocol,report,archive):
    """An executed ray fan and one actual ±DPH response, without a fitted curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z=archive["baseline_z_mm"];alive=archive["baseline_alive"]
    live=np.flatnonzero(np.all(alive,axis=0) & (archive["weights"]>0))
    selected=live[np.linspace(0,len(live)-1,min(80,len(live))).astype(int)] if len(live) else []
    figure,(ax,response)=plt.subplots(2,1,figsize=(13,8),layout="constrained")
    ax.plot(z,archive["baseline_x_m"][:,selected]*1e3,linewidth=.65,alpha=.55)
    ax.set(xlabel="Z (mm)",ylabel="Laboratory X (mm)",title="Actual CUDA baseline: same Tip rays, full Cs and hexapoles")
    for lens in state.lenses:
        if lens.enabled and z[0]<lens.z_mm<z[-1]+.01:
            ax.axvline(lens.z_mm,color=".5",alpha=.4,linewidth=.6)
            ax.text(lens.z_mm,.98,lens.name,transform=ax.get_xaxis_transform(),rotation=90,
                    va="top",ha="right",fontsize=7)
    for component in (state.hp2_hexapole,state.hp1_hexapole):
        if component.enabled:
            ax.axvline(component.z_mm,color="purple",alpha=.5,linewidth=.8)
            ax.text(component.z_mm,.98,component.name,transform=ax.get_xaxis_transform(),rotation=90,
                    va="top",ha="right",fontsize=7,color="purple")
    for row in report["baseline"]["source_crossovers"]:
        ax.axvspan(*row["interval_mm"],alpha=.07,color="green")
        for refinement in row["refinements"]:
            if not refinement["crossovers"] and "z_mm" in refinement:
                ax.axvline(refinement["z_mm"],color="tab:orange",linestyle=":",linewidth=1.2)
            for waist in refinement["crossovers"]:
                ax.axvline(waist["z_mm"],color="magenta",linestyle="--",linewidth=1.)
    key=COMPONENT_KEYS[0]
    baseline=archive[key+"_baseline"]
    for label,color,command in (("a0_s1_-1","tab:blue","−X"),("a0_s1_+1","tab:red","+X")):
        values=archive[key+"_"+label]-baseline
        selected=archive[key+"_common"]
        response.scatter(values[0,selected]*1e9,values[1,selected]*1e9,s=5,alpha=.4,color=color,
                         label=f"{command} {protocol['perturbation_mrad']:g} mrad")
    response.set(xlabel="Change of exit X (nm)",ylabel="Change of exit Y (nm)",
        title="DPH2: same-ray displacement at the finite coil exit relative to baseline")
    response.legend();response.grid(alpha=.2);ax.grid(alpha=.15)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([],[],color="magenta",linestyle="--",label="Executed 2D waist"),
        Line2D([],[],color="tab:orange",linestyle=":",label="Radial minimum; approximate common waist")],
        loc="lower left",fontsize=8)
    figure.savefig(output,dpi=170);plt.close(figure)


def execute(args,state,snapshot,protocol,report):
    from temsim.physics.backend_execution import backend_receipts
    from temsim.optics.electron_gun.source import trace_source_to_exit
    from temsim.physics.compute_backend import choose_ray_backend
    report["status"]="RUNNING"
    save_report(args.output/"report.json",report)
    selected,reason=choose_ray_backend("Require GPU",acceleration_enabled=True,ray_count=args.rays)
    if selected!="CUDA GPU" or reason:raise RuntimeError("CUDA is required before physical execution")
    receipts=report.setdefault("backend_receipts",[])
    rows=[]
    with backend_receipts(rows):gun=trace_source_to_exit(state)
    receipts.extend(rows)
    report["source_exit_identity"]=source_identity(gun.exit_bundle)
    report["source_parameters"]=state.electron_gun.to_dict()
    coarse,prefix,baseline_report=baseline(state,protocol,gun,receipts)
    report["baseline"]=baseline_report
    points,masks,_,_,_=coarse
    weights=np.asarray(gun.exit_bundle.weight)
    archive={"weights":weights,"ray_id":np.asarray(gun.exit_bundle.ray_id),"baseline_z_mm":points.z_mm,"baseline_alive":masks}
    for name in ("x_m","y_m","tx_rad","ty_rad","kinetic_energy_ev","flight_time_s"):
        archive["baseline_"+name]=np.asarray(getattr(points,name))
    archive["baseline_stops_mm"]=coarse[3]
    archive["baseline_stop_keys"]=np.asarray(coarse[4],dtype=str)
    np.savez_compressed(args.output/"coordinates.npz",**archive)
    save_report(args.output/"report.json",report)
    for component,row in zip(components(state),protocol["components"]):
        print(f"Local CUDA finite-field validation: {component.key}",flush=True)
        start,stop=row["local_interval_mm"];entry,coordinates=phase_points(points,start)
        initial=resumed_initial(points,masks,start,coarse[3],coarse[4])
        saved=(component.kick_x_mrad,component.kick_y_mrad,component.enabled)
        cases={};case_alive={}
        def run(label,x,y,enabled=True):
            component.kick_x_mrad=x;component.kick_y_mrad=y;component.enabled=enabled
            result=trace(state,start,stop,**initial,planes=(stop,),step=args.step,receipts=receipts,label=component.key+":"+label)
            index,phase=phase_points(result[0],stop)
            cases[label]=phase;case_alive[label]=result[1][index]
        try:
            run("baseline",saved[0],saved[1])
            for axis in range(2):
                for scale in (1.,.5):
                    for sign in (-1.,1.):
                        command=np.array(saved[:2]);command[axis]+=sign*scale*args.delta_mrad
                        run(f"a{axis}_s{scale:g}_{sign:+g}",*command)
            run("disabled_baseline",saved[0],saved[1],False)
            run("disabled_command",saved[0]+args.delta_mrad,saved[1]-args.delta_mrad,False)
        finally:
            component.kick_x_mrad,component.kick_y_mrad,component.enabled=saved
        for key,value in cases.items():
            archive[component.key+"_"+key]=value
            archive[component.key+"_"+key+"_alive"]=case_alive[key]
        np.savez_compressed(args.output/"coordinates.npz",**archive)
        save_report(args.output/"report.json",report)
        active_labels=[k for k in cases if not k.startswith("disabled")]
        common=np.logical_and.reduce([case_alive[k] for k in active_labels]) & masks[entry] & (weights>0)
        fraction=float(weights[common].sum()/weights[masks[entry] & (weights>0)].sum())
        ref_indices=select_reference_rays(coordinates,initial["energy"],weights,common,args.reference_rays)
        reference,tangent,ref_evidence=reference_for_component(state,row,coordinates[:,ref_indices],initial["energy"][ref_indices],args.step,2e-10)
        _,tangent_fine,ref_fine_evidence=reference_for_component(state,row,coordinates[:,ref_indices],initial["energy"][ref_indices],args.step,1e-11)
        length=row["effective_length_mm"]*1e-3
        ref_change=relative_response_error(tangent,tangent_fine,weights[ref_indices],length)
        disabled=np.array_equal(cases["disabled_baseline"],cases["disabled_command"],equal_nan=True) and np.array_equal(case_alive["disabled_baseline"],case_alive["disabled_command"])
        axis_results=[]
        for axis in range(2):
            h=args.delta_mrad*1e-3
            plus,minus=cases[f"a{axis}_s1_+1"],cases[f"a{axis}_s1_-1"]
            plus_half,minus_half=cases[f"a{axis}_s0.5_+1"],cases[f"a{axis}_s0.5_-1"]
            derivative=(plus-minus)/(2*h);half=(plus_half-minus_half)/h
            linear=relative_response_error(derivative[:,common],half[:,common],weights[common],length)
            # Normalize the even residual by the actual odd drive response.
            even=relative_response_error(derivative[:,common]+((plus+minus)/2-cases["baseline"])[:,common]/h,
                derivative[:,common],weights[common],length)
            independent=relative_response_error(half[:,ref_indices],tangent_fine[:,axis],weights[ref_indices],length)
            axis_results.append(dict(axis="XY"[axis],linear=linear,even=even,independent=independent,
                weighted_response_x_y_tx_ty=(derivative[:,common]@(weights[common]/weights[common].sum())).tolist(),
                passed=bool(linear["relative_error"]<=LIMITS["relative_linear_response"] and
                    even["relative_error"]<=LIMITS["relative_even_response"] and
                    independent["relative_error"]<=LIMITS["relative_independent_response"] and
                    independent["polarity_cosine"]>0 and independent["reference_response_norm"]>LIMITS["minimum_response_norm"])))
        component_result=dict(**row,common_rays=int(common.sum()),common_current_fraction=fraction,
            independent_reference_indices=ref_indices.tolist(),reference=ref_evidence,reference_fine=ref_fine_evidence,
            reference_refinement=ref_change,disabled_zero_response=disabled,axes=axis_results,
            reference_scope="Actual selected baseline entrance rays, preserving individual energy. Independent integration/tangents of shared physical field providers; not an independently measured field.",
            passed=bool(disabled and fraction>=LIMITS["minimum_common_current_fraction"] and
                ref_change["relative_error"]<=LIMITS["reference_refinement_change"] and all(a["passed"] for a in axis_results)))
        report.setdefault("components",[]).append(component_result)
        for key,value in cases.items():
            archive[component.key+"_"+key]=value
        archive[component.key+"_reference_indices"]=ref_indices
        archive[component.key+"_reference_baseline"]=reference
        archive[component.key+"_reference_tangent"]=tangent_fine
        archive[component.key+"_common"]=common
        np.savez_compressed(args.output/"coordinates.npz",**archive)
        save_report(args.output/"report.json",report)
    snapshot.verify_current_inputs(state)
    np.savez_compressed(args.output/"coordinates.npz",**archive)
    render_figure(args.output/"corrector_particle_validation.png",state,protocol,report,archive)
    report["deflector_validation_status"]="PASS" if all(row["passed"] for row in report["components"]) else "FAIL"
    report["baseline_validation_status"]="PASS" if all(baseline_report["checks"].values()) else "FAIL"
    report["status"]="PASS" if report["deflector_validation_status"]==report["baseline_validation_status"]=="PASS" else "FAIL"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--inputs",type=Path,help="Optional current-implementation instrument snapshot")
    parser.add_argument("--before-manifests",type=Path,help="Optional directory with saved before_*.toml column manifests")
    parser.add_argument("--rays",type=int,default=3000)
    parser.add_argument("--step",type=float,default=.05)
    parser.add_argument("--delta-mrad",type=float,default=.001)
    parser.add_argument("--reference-rays",type=int,default=32)
    parser.add_argument("--execute",action="store_true",help="Run GPU/source validation; omitted means prepare only")
    args=parser.parse_args()
    if args.rays<5 or args.reference_rays<5 or not np.isfinite((args.step,args.delta_mrad)).all() or args.step<=0 or args.delta_mrad<=0:
        parser.error("Ray counts must be at least five, step and perturbation finite and positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new empty output directory")
    args.output.mkdir(parents=True,exist_ok=True)
    report=dict(status="PREPARING",started_utc=datetime.now(timezone.utc).isoformat())
    started=perf_counter()
    try:
        state,snapshot,protocol=prepare(args)
        report["manifest_comparison"]=manifest_comparison(args.before_manifests)
        save_report(args.output/"inputs.json",snapshot.to_dict())
        save_report(args.output/"protocol.json",protocol)
        report.update(input_identity=snapshot.digest,protocol=protocol,status="PREPARED_NOT_EXECUTED")
        if args.execute:execute(args,state,snapshot,protocol,report)
    except Exception as exc:
        report.update(status="FAILED",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_s"]=perf_counter()-started
        save_report(args.output/"report.json",report)
        print(json.dumps(dict(status=report["status"],report=str((args.output/"report.json").resolve()))),flush=True)
    return 0 if report["status"] in ("PREPARED_NOT_EXECUTED","PASS") else 2


if __name__=="__main__":
    raise SystemExit(main())

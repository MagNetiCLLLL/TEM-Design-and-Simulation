"""Convergence gates compare frozen physics and retain numerical failures."""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import pytest

spec=spec_from_file_location("particle_count_convergence",Path(__file__).resolve().parents[1]/"scripts/validate_particle_count_convergence.py")
module=module_from_spec(spec)
spec.loader.exec_module(module)


def report(count, **changes):
    point=dict(alpha95_mrad=10.,rms_radius_nm=.2,diameter95_nm=.6,current_a=1e-5,local_focus_offset_nm=0.)
    point.update(changes)
    return dict(rays=count,step_mm=.05,physical_input_signature="frozen",specimen=point,
                execution_identity=dict(solver="fixed",external_inputs="fixed"))


def test_only_numerical_counts_and_column_step_are_excluded_from_physical_identity():
    base=dict(step_mm=.05,electron_gun=dict(ray_count=193,emission_energy_ev=.3,energy_spread_fwhm_ev=.3),
              lens=dict(percent=30.,z_mm=500.,length_mm=10.),gun_trace_step_mm=.2)
    numerical=deepcopy(base);numerical["step_mm"]=.025;numerical["electron_gun"]["ray_count"]=3073
    assert module.physical_input_signature(base)==module.physical_input_signature(numerical)
    for field in ("percent","z_mm","length_mm"):
        changed=deepcopy(numerical);changed["lens"][field]+=1.
        assert module.physical_input_signature(base)!=module.physical_input_signature(changed)
    changed=deepcopy(numerical);changed["electron_gun"]["energy_spread_fwhm_ev"]=.31
    assert module.physical_input_signature(base)!=module.physical_input_signature(changed)


def test_angle_uses_nonzero_finer_population_scale_and_focus_absolute_budget():
    a=report(769);b=report(3073,alpha95_mrad=10.03,local_focus_offset_nm=.26)
    original=deepcopy((a,b))
    result=module.compare_counts(a,b)
    assert not result["passed"]
    assert result["metrics"]["alpha95_mrad"]["relative_change"]==pytest.approx(.03/10.03)
    assert not result["metrics"]["alpha95_mrad"]["passed"]
    assert not result["metrics"]["local_focus_offset_nm"]["passed"]
    assert (a,b)==original
    assert module.compare_counts(a,report(3073,alpha95_mrad=10.01,local_focus_offset_nm=.1))["passed"]


def test_real_current_and_size_changes_cannot_be_hidden_or_nonfinite_promoted():
    a=report(769);b=report(3073,current_a=.9e-5,rms_radius_nm=.25,local_focus_offset_nm=None)
    result=module.compare_counts(a,b)
    assert not result["passed"]
    assert not result["metrics"]["current_a"]["passed"]
    assert not result["metrics"]["rms_radius_nm"]["passed"]
    assert not result["metrics"]["local_focus_offset_nm"]["passed"]
    assert not module.compare_counts(a,report(3073,alpha95_mrad=0.))["passed"]
    assert not module.compare_counts(a,report(3073,alpha95_mrad=None))["passed"]
    import numpy as np
    from types import SimpleNamespace
    cp=SimpleNamespace(**{key:np.zeros((1,2)) for key in
        ("x_m","y_m","tx_rad","ty_rad","flight_time_s","kinetic_energy_ev")})
    cp.x_m[0,1]=np.nan
    with pytest.raises(ValueError,match="Nonfinite"):
        module.require_finite_population(cp,np.ones((1,2),bool),np.ones(2))
    module.require_finite_population(cp,np.array([[True,False]]),np.ones(2))


def test_mixed_physics_steps_or_nonincreasing_counts_cannot_claim_count_convergence():
    a=report(769)
    for key,value in (("physical_input_signature","changed"),("step_mm",.025),("rays",193)):
        b=report(3073);b[key]=value
        with pytest.raises(ValueError):
            module.compare_counts(a,b)
    b=report(3073);b["execution_identity"]["solver"]="different"
    result=module.compare_counts(a,b)
    assert result["numerical_metrics_passed"] and not result["passed"]
    assert result["execution_identity_status"]=="MISSING_OR_DIFFERENT"
    b=report(3073);b.pop("execution_identity")
    assert not module.compare_counts(a,b)["passed"]

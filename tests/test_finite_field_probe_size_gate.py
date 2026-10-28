"""Declared finite-field probe-size acceptance; no physical propagation."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import math
import pytest

spec=spec_from_file_location('probe_size_gate_cli',Path(__file__).resolve().parents[1]/'scripts/qualify_finite_field_nanoprobe.py')
qualification=module_from_spec(spec)
spec.loader.exec_module(qualification)


def failures(diameter, **kwargs):
    measurement=SimpleNamespace(accepts=lambda _:True,
        statistics=SimpleNamespace(radius_95_m=diameter/(2e9)))
    topology={'passed':True,'direct_focus':{'status':'PASS'}}
    corrector={'waist_evidence_passed':True,'angle_matching_passed':True}
    return qualification.qualification_failures(topology,measurement,corrector,30.,**kwargs)


@pytest.mark.parametrize('diameter,passes',[(.699999999,True),(.7,False),(.700000001,False),(5.,False),(0.,True)])
def test_default_size_gate_is_strictly_below_point_seven_nm(diameter,passes):
    assert (not failures(diameter)) is passes


@pytest.mark.parametrize('diameter',[math.nan,math.inf,-math.inf,-.1])
def test_invalid_measured_diameter_cannot_qualify(diameter):
    assert failures(diameter)


@pytest.mark.parametrize('limit',[0.,-.1,math.nan,math.inf,-math.inf,None,True,'invalid'])
def test_invalid_declared_limit_is_rejected_before_measurement(limit):
    with pytest.raises(ValueError,match='diameter'):
        failures(.1,maximum_diameter95_nm=limit)


def test_explicit_alternative_limit_is_reportable_and_remains_strict():
    assert failures(1.)
    assert not failures(1.,maximum_diameter95_nm=2.)
    assert failures(2.,maximum_diameter95_nm=2.)

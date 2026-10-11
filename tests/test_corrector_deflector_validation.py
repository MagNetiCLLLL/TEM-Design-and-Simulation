"""Analytical checks for the independent validation ODE, not another ray engine."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import elementary_charge
from scipy.linalg import expm


SCRIPTS=Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0,str(SCRIPTS))
try:
    spec=importlib.util.spec_from_file_location("corrector_validation",SCRIPTS/"validate_corrector_deflectors.py")
    validation=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation)
finally:
    sys.path.remove(str(SCRIPTS))


def test_independent_uniform_finite_coil_has_exact_length_energy_and_sign():
    p=np.array([2.,3.,4.])*1e-22
    initial=np.array([[1.,-2.,3.],[2.,3.,4.],[.1,.2,-.1],[.2,-.1,.3]])*1e-5
    start,end=.2,.23
    support=(.204,.216)
    constant=lambda z,mid:(0.,)*8
    final,tangent,_=validation.independent_reference(initial,p,3e-22,(start,end),constant,support,
                                                     command_rad=(2e-6,-3e-6))
    ratio=3e-22/p
    lever=end-sum(support)/2
    expected=np.zeros((4,2,3))
    expected[0,0]=expected[1,1]=ratio*lever
    expected[2,0]=expected[3,1]=ratio
    assert_allclose(tangent,expected,rtol=2e-13,atol=1e-16)
    free=initial.copy();free[:2]+=initial[2:]*(end-start)
    assert_allclose(final,free+np.einsum("ijn,j->in",expected,[2e-6,-3e-6]),rtol=2e-13,atol=1e-16)


def test_independent_solenoid_tangent_matches_augmented_matrix_exponential():
    p=3e-22;b=.07;g=-elementary_charge*b/(2*p);length=.008
    matrix=np.array([[0,g,1,0],[-g,0,0,1],[-g*g,0,0,g],[0,-g*g,-g,0]],float)
    augmented=np.zeros((6,6));augmented[:4,:4]=matrix
    augmented[2,4]=augmented[3,5]=1/length
    transform=np.eye(4);transform[2,1]=g;transform[3,0]=-g
    expected=transform@expm(augmented*length)[:4,4:]
    _,tangent,_=validation.independent_reference(np.zeros((4,1)),np.array([p]),p,(0,length),
        lambda z,mid:(b,0,0,0,0,0,0,0),(0,length))
    assert_allclose(tangent[:,:,0],expected,rtol=2e-10,atol=1e-12)


def test_nonlinear_hexapole_and_cs_tangent_matches_independent_command_difference():
    initial=np.array([[1e-4],[-2e-4],[.001],[.002]])
    args=(initial,np.array([3e-22]),3e-22,(0,.01),
          lambda z,mid:(.005,30.,-30.,4.,2e5,-3e4,0,0),(.002,.008))
    kwargs=dict(cs_events=((.006,2e5),),rtol=1e-11)
    _,tangent,_=validation.independent_reference(*args,**kwargs)
    for axis in (0,1):
        command=np.zeros(2);command[axis]=1e-7
        plus,_,_=validation.independent_reference(*args,command_rad=command,**kwargs)
        minus,_,_=validation.independent_reference(*args,command_rad=-command,**kwargs)
        assert_allclose((plus-minus)/2e-7,tangent[:,axis],rtol=5e-8,atol=1e-10)


def test_response_metric_rejects_reversed_polarity_and_uses_si_scale():
    response=np.array([[.01,.02],[.02,-.01],[1.,2.],[2.,-1.]])
    same=validation.relative_response_error(response,response,np.array([1.,2.]),.01)
    reverse=validation.relative_response_error(-response,response,np.array([1.,2.]),.01)
    assert same["relative_error"]==0
    assert_allclose(same["polarity_cosine"],1)
    assert_allclose(reverse["relative_error"],2)
    assert_allclose(reverse["polarity_cosine"],-1)


def test_resuming_does_not_reuse_future_baseline_stop():
    from types import SimpleNamespace
    cp=SimpleNamespace(z_mm=np.array([1.]),x_m=np.zeros((1,2)),y_m=np.zeros((1,2)),
        tx_rad=np.zeros((1,2)),ty_rad=np.zeros((1,2)),kinetic_energy_ev=np.full((1,2),3e5),
        flight_time_s=np.ones((1,2)))
    initial=validation.resumed_initial(cp,np.array([[True,False]]),1.,np.array([2.,.5]),["future","past"])
    assert np.isnan(initial["stops"][0])
    assert initial["stops"][1]==.5
    assert initial["keys"]==["","past"]


def test_electric_reference_retains_longitudinal_acceleration_in_response():
    from scipy.constants import electron_mass,speed_of_light
    from scipy.integrate import quad
    energy=3e5;gradient=1e6;end=.01;lo,hi=.002,.008
    def momentum(z):
        kinetic=(energy+gradient*z)*elementary_charge
        return np.sqrt(kinetic*(kinetic+2*electron_mass*speed_of_light**2))/speed_of_light
    class UniformAxialElectric:
        def interpolate(self,xyz):
            ef=np.zeros_like(xyz);ef[:,2]=-gradient
            return 3e5+gradient*xyz[:,2],ef
    _,response,receipt=validation.independent_electric_reference(np.zeros((4,1)),np.array([energy]),
        momentum(0),(0,end),lambda z,mid:(0.,)*8,(lo,hi),UniformAxialElectric(),rtol=1e-11)
    position=quad(lambda z:momentum(0)/momentum(z)*min(max(z-lo,0),hi-lo)/(hi-lo),
                  0,end,points=[lo,hi],epsabs=1e-14)[0]
    expected=np.array([[position,0],[0,position],[momentum(0)/momentum(end),0],[0,momentum(0)/momentum(end)]])
    assert_allclose(response[:,:,0],expected,rtol=1e-10,atol=1e-12)
    assert receipt["maximum_sampled_energy_change_ev"]>=9999

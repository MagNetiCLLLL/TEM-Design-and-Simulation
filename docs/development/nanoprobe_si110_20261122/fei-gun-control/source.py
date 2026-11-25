"""Prescribed 15 microampere emission with a Murphy-Good surface-current law."""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import qmc

# Fixed equal-count importance strata in the *physical* cumulative current.
# Aperture survivors may occupy a very small central fraction. Every initial
# area and every particle carries its original positive current throughout.
STRATA = np.array([0., 1/65536, 1/4096, 1/256, 1/32, .125, .375, .75, 1.])
REST_EV = 510998.95069


def distribution(apex_field, radius_nm=100., phi=4.35):
    angle = np.linspace(0., np.deg2rad(80.), 32769)
    field = apex_field*np.cos(angle)
    f = 1.439965*field/phi**2
    if np.max(f)>=1 or np.min(f)<=0:
        raise ValueError('A tunnelling source requires a positive barrier')
    nu = 1-f+f*np.log(f)/6
    t = 1+f/9-f*np.log(f)/18
    # Biswas (2018), eq. 4. J in microamperes per square nm, F in V/nm.
    density = 1.541434/phi * (field/t)**2 * np.exp(-6.830890*phi**1.5*nu/field)
    # Exact area element of this model's SPHERICAL cap, with the smooth-tip
    # cosine-law field; the paper's parabolic surface has a different Jacobian.
    cumulative = cumulative_trapezoid(density*2*np.pi*radius_nm**2*np.sin(angle), angle, initial=0)
    return angle, cumulative/cumulative[-1], float(cumulative[-1])


def emit(apex_field, count):
    if count < 1024 or count % (len(STRATA) - 1):
        raise ValueError('Use at least 1024 original particles in complete eight-stratum blocks')
    angle, cdf, raw_current = distribution(apex_field)
    strata = len(STRATA)-1
    row = np.arange(count)
    phase = np.zeros((count,6))
    envelope = np.zeros(count)
    budget = np.zeros(count)
    for k,(lo,hi) in enumerate(zip(STRATA[:-1],STRATA[1:])):
        chosen = row[k::strata]
        power = (len(chosen) - 1).bit_length()
        u = qmc.Sobol(4, scramble=True, seed=20261123+k).random_base2(power)[:len(chosen)]
        theta = np.interp(lo+(hi-lo)*u[:,0],cdf,angle)
        az = 2*np.pi*u[:,1]
        st,ct,ca,sa = np.sin(theta),np.cos(theta),np.cos(az),np.sin(az)
        n = np.column_stack((st*ca,st*sa,ct))
        tangent = np.column_stack((ct*ca,ct*sa,-st))
        around = np.column_stack((-sa,ca,np.zeros_like(sa)))
        # Exponential transverse energy from the linearized cold free-electron
        # barrier (d_F/t_F), independent uniform azimuth. The 0.2 eV outward
        # energy is the model's post-emission launch convention.
        field = apex_field*ct
        f = 1.439965*field/4.35**2
        t = 1+f/9-f*np.log(f)/18
        scale_ev = 2*field/(3*6.830890*np.sqrt(4.35)*t)
        transverse = -scale_ev*np.log1p(-u[:,2])
        az_t = 2*np.pi*u[:,3]
        energy = .2+transverse
        direction = np.sqrt(.2) * n + np.sqrt(transverse)[:,None] * (
            np.cos(az_t)[:,None]*tangent + np.sin(az_t)[:,None]*around)
        phase[chosen,:3] = 1e-7*n
        phase[chosen,2] = -2e-7*np.sin(theta/2)**2
        phase[chosen,3:] = direction*np.sqrt(2/REST_EV+energy[:,None]/REST_EV**2)
        envelope[chosen] = theta
        budget[chosen] = 15*(hi-lo)/len(chosen)
    assert np.all(budget > 0) and np.isclose(budget.sum(),15)
    assert np.unique(phase,axis=0).shape[0] == count
    return phase, budget, envelope, dict(apex_V_per_nm=apex_field,work_function_eV=4.35,
        cap_degrees=80,raw_Murphy_Good_uA=raw_current,prescribed_uA=15.,
        polar_quantiles_deg=np.rad2deg(np.interp([.5,.9,.95,.99],cdf,angle)).tolist(),
        current_fraction_above_60_deg=float(1-np.interp(np.pi/3,angle,cdf)),
        strata_cumulative_current=STRATA.tolist())

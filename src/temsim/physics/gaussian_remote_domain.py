"""Conservative binary64 remote-domain admission for retained Gaussian A_phi.

No threshold is applied to a sampled field. Every exact continuation and every
retained polynomial term, including Cartesian A derivatives through order two,
must be bounded below half the smallest binary64 subnormal in SI units. The
original polynomial is then evaluated unchanged. This is not an OEM far-field
model or an enlargement of the native near-axis mechanical bore.
"""
import math

import numpy as np


POLICY = "gaussian-binary64-remote-roundoff-v1"


def metadata():
    return {"policy":POLICY,"native_radial_domain":"unchanged",
        "qualification":"exact Gaussian vacuum continuation and retained finite A polynomial with Cartesian derivatives through order 2",
        "absolute_bound":"strictly below 2^-1075 per SI-valued combined Frobenius norm",
        "arithmetic":"outward binary64 basic operations; frexp integer exponents; exp(-alpha) < 2^-alpha",
        "field_evaluation":"unchanged retained polynomial; no coefficient or tail is replaced",
        "limits":"Gaussian analytic engineering model only; no FEM, material or measured-field extrapolation"}


def _up(value):
    return np.nextafter(value,np.inf)


def _down(value):
    return np.nextafter(value,0.)


def _add(a,b):
    return _up(np.asarray(a)+b)


def _mul(a,b):
    return _up(np.asarray(a)*b)


def _power(value,order):
    result=np.ones_like(value)
    for _ in range(order):
        result=_mul(result,value)
    return result


def remote_certificate(native,local_positions_m,radial_order,*,_z_bounds=None):
    """Return pointwise admission and conservative integer-bound evidence.

    Coordinates are the same evaluated local binary64 coordinates as the
    retained provider. Basic operations are rounded outward. No representable
    ``2**-1075`` is formed. Nonfinite prefactors or unresolved underflow in a
    denominator reject admission instead of turning Inf*0 into a field zero.
    """
    from temsim.physics.lens_field_provider import FrozenAnalyticField
    if type(native) is not FrozenAnalyticField or type(radial_order) is not int or radial_order not in (1,3,5,7,9):
        raise ValueError("Remote qualification requires a retained native Gaussian round potential")
    local=np.asarray(local_positions_m)
    if local.dtype.kind not in "iuf" or local.shape[-1:]!=(3,) or not np.isfinite(local).all():
        raise ValueError("Remote qualification coordinates must be finite real local XYZ")
    local=np.asarray(local,dtype=float)
    count=(radial_order+1)//2
    terms=tuple(term for term in native.terms_t_m if term[0]!=0.)
    # A zero current does not grant a new near-field spatial domain.
    if not terms:
        return {"admitted":np.zeros(local.shape[:-1],bool),"minimum_binary_margin":np.full(local.shape[:-1],-np.inf)}
    extra=(len(terms)*count-1).bit_length()
    rotation=np.abs(native.registration.rotation_array)
    row_sum=_add(_add(rotation[:,0],rotation[:,1]),rotation[:,2])
    rotation_factor=_mul(8.,_power(np.asarray(max(1.,float(np.max(row_sum)))),3))
    admitted=np.ones(local.shape[:-1],bool)
    margin=np.full(local.shape[:-1],np.inf)
    with np.errstate(over="ignore",under="ignore",invalid="ignore",divide="ignore"):
        r2=_add(_mul(abs(local[...,0]),abs(local[...,0])),_mul(abs(local[...,1]),abs(local[...,1])))
        radius=_up(np.sqrt(r2))
        for amplitude,centre,sigma in terms:
            if _z_bounds is None:
                delta=abs(local[...,2]-centre)
                d_lower=_down(delta)
                d_upper=_up(delta)
            else:
                low,high=_z_bounds
                d_lower=_down(max(0.,low-centre,centre-high))
                d_upper=_up(max(abs(low-centre),abs(high-centre)))
            numerator=np.nextafter(_down(d_lower*d_lower)-r2,-np.inf)
            denominator=_up(2.*_up(sigma*sigma))
            alpha=_down(numerator/denominator)
            inverse=_up(1./sigma)
            inverse2=_mul(inverse,inverse)
            # The Cartesian disk-average representation has |direction|<=1.
            g1=_mul(_add(d_upper,radius),inverse2)
            g2=_add(_mul(g1,g1),inverse2)
            exact=np.maximum.reduce((radius/2.,_add(1.,_mul(radius,g1)),_add(g1,_mul(radius,g2))))
            exact=_mul(_mul(abs(amplitude),rotation_factor),exact)
            u=_mul(d_upper,inverse)
            hermite=[np.ones_like(u),u]
            for order in range(2,radial_order+2):
                hermite.append(_add(_mul(u,hermite[-1]),_mul(order-1,hermite[-2])))
            for n in range(count):
                degree=2*n+1
                factor=_up((2.**n)/(2.**(2*n+1)*math.factorial(n)*math.factorial(n+1)))
                finite=np.zeros_like(radius)
                for xy in range(min(2,degree)+1):
                    for axial in range(3-xy):
                        order=2*n+axial
                        polynomial=_mul(float(degree**xy),_power(radius,degree-xy))
                        derivative=_mul(hermite[order],_power(np.full_like(radius,inverse),order))
                        finite=np.maximum(finite,_mul(_mul(polynomial,derivative),factor))
                finite=_mul(_mul(abs(amplitude),rotation_factor),finite)
                bound=np.maximum(exact,finite)
                exponent=np.frexp(bound)[1].astype(np.int64)
                required=1075+extra+exponent
                valid=(np.isfinite(alpha)&(alpha>0)&np.isfinite(bound)&(bound>0)
                       &np.isfinite(denominator)&(denominator>0)&(alpha>=required))
                admitted &= valid
                margin=np.minimum(margin,np.where(valid,alpha-required,-np.inf))
    return {"admitted":admitted,"minimum_binary_margin":margin}


def require_remote_domain(native,local_positions_m,radial_order):
    if not np.all(remote_certificate(native,local_positions_m,radial_order)["admitted"]):
        raise ValueError("Position outside native round-lens domain lacks Gaussian remote-roundoff qualification")


def require_global_box(native,global_points_m,radial_order,valid_radius_m):
    """Certify the entire global axis-aligned enclosing box, not only corners.

    The cylinder/remote union is nonconvex. Interval transforms and each
    Gaussian's minimum axial distance prevent two remote end planes from
    admitting a box that crosses an unqualified near-field region.
    """
    points=np.asarray(global_points_m)
    if points.dtype.kind not in "iuf" or points.shape[-1:]!=(3,) or not points.size or not np.isfinite(points).all():
        raise ValueError("Round-lens swept box requires finite global XYZ points")
    points=np.asarray(points,dtype=float).reshape(-1,3)
    low=np.nextafter(points.min(axis=0)-native.registration.origin_array_m,-np.inf)
    high=np.nextafter(points.max(axis=0)-native.registration.origin_array_m,np.inf)
    rotation=native.registration.rotation_array
    local_low,local_high=np.zeros(3),np.zeros(3)
    with np.errstate(over="ignore",under="ignore",invalid="ignore",divide="ignore"):
        for axis in range(3):
            for dimension in range(3):
                a,b=low[axis]*rotation[axis,dimension],high[axis]*rotation[axis,dimension]
                local_low[dimension]=np.nextafter(local_low[dimension]+np.nextafter(min(a,b),-np.inf),-np.inf)
                local_high[dimension]=np.nextafter(local_high[dimension]+np.nextafter(max(a,b),np.inf),np.inf)
        transverse=np.maximum(abs(local_low[:2]),abs(local_high[:2]))
        radius=_up(np.sqrt(_add(_mul(transverse[0],transverse[0]),_mul(transverse[1],transverse[1]))))
    if not np.isfinite((*local_low,*local_high,radius)).all():
        raise ValueError("Round-lens swept box exceeds finite interval arithmetic")
    if radius<=valid_radius_m:
        return
    certificate=remote_certificate(native,np.array([radius,0.,0.]),radial_order,
        _z_bounds=(local_low[2],local_high[2]))
    if not certificate["admitted"]:
        raise ValueError("Round-lens swept box crosses a region without Gaussian remote-roundoff qualification")

"""CPU reference transport for imported vector fields in the column.

Coordinates are metres and mechanical slopes dx/dz, dy/dz.  Unmapped round
lenses retain the existing paraxial canonical equations.  Imported fields
enter once through q/p * sqrt(1+tx**2+ty**2) times the full Lorentz slope
equations, evaluated at each ray's actual XYZ at every RK stage.  The magnetic
term uses constant relativistic momentum magnitude: a static B does no work.
Z is the independent variable, so this solver covers forward column rays;
turning trajectories must be handled by a time-domain particle solver.
"""

import numpy as np

from temsim.physics.ray_integrator import canonical_rk4_step, canonical_rk4_step_with_time


def magnetic_slope_derivative(fields, positions, slopes, charge_over_p, *, xp=np):
    """Sum genuine vector fields and lens-local paraxial analytical laws.

    A nonzero placement must not silently add full-Lorentz cubic terms to a
    lens whose unposed model is paraxial with a separately configured Cs.
    """
    result = xp.zeros_like(slopes)
    magnetic = xp.zeros_like(positions)
    for field in fields:
        paraxial = getattr(field, "slope_derivative", None)
        if callable(paraxial):
            result += paraxial(positions, slopes, charge_over_p)
        else:
            magnetic += field.field_at_global_positions_t(positions)
    ux, uy = slopes.T
    bx, by, bz = magnetic.T
    factor = charge_over_p*xp.sqrt(1.+ux*ux+uy*uy)
    result[:, 0] += factor*(uy*bz-(1.+ux*ux)*by+ux*uy*bx)
    result[:, 1] += factor*((1.+uy*uy)*bx-ux*bz-ux*uy*by)
    return result


def vector_map_rk4(
    kx, ky, hn, hs, larmor_axis, inverse_momentum, cs_kick,
    thin_power, thin_rotation, step_m, x, tx, y, ty,
    kickx, kicky, save_index, checkpoint_index, kxy,
    dipole_bx_t, dipole_by_t, reference_momentum, *, z_mm, mapped_fields,
    defer_nonfinite_until_clipping=False,
    step_operator=None,
    initial_time_s=None, inverse_speed=None,
    posed_spherical_kicks=None,
    xp=np, cancel_check=None,
):
    nr, ns, nc = x.size, save_index.size, checkpoint_index.size
    # Finite-difference transfer Jacobians need float64 even in saved history.
    X, TX, Y, TY = (xp.empty((ns, nr), np.float64) for _ in range(4))
    CX, CTX, CY, CTY = (xp.empty((nc, nr), np.float64) for _ in range(4))
    saved = captured = 0
    time = xp.array(initial_time_s, dtype=np.float64, copy=True) if initial_time_s is not None else None
    T = xp.empty((ns, nr), np.float64) if time is not None else None
    CT = xp.empty((nc, nr), np.float64) if time is not None else None
    charge_over_p = -1.602176634e-19 * inverse_momentum
    magnetic_scale = reference_momentum[0] * inverse_momentum
    larmor_values = xp.asarray(larmor_axis)
    supports = tuple((item, *item.field_support_mm)
                     for item in mapped_fields if item.scale != 0.0)
    from temsim.physics.intrinsic_aberrations import FrozenIntrinsicAberrationKick
    for j in range(step_m.size + 1):
        if cancel_check is not None and j % 64 == 0:
            cancel_check()
        power = thin_power[j] * magnetic_scale**2
        tx, ty = tx - power * x, ty - power * y
        if thin_rotation[j] != 0.0:
            angle = thin_rotation[j] * magnetic_scale
            co, si = xp.cos(angle), xp.sin(angle)
            x, y = co * x - si * y, si * x + co * y
            tx, ty = co * tx - si * ty, si * tx + co * ty
        tx, ty = tx + kickx[j], ty + kicky[j]
        radial = cs_kick[j] * (x*x + y*y)
        tx, ty = tx - radial*x, ty - radial*y
        for kick in (posed_spherical_kicks or {}).get(j, ()):
            if isinstance(kick, FrozenIntrinsicAberrationKick):
                x, tx, y, ty = kick.apply_canonical(x, tx, y, ty,
                    momentum_kg_m_s=1./inverse_momentum,
                    reference_momentum_kg_m_s=float(reference_momentum[0]),
                    fields=mapped_fields, aligned_bz_t=-2.*larmor_axis[2*j]/1.602176634e-19)
                continue
            # Preserve the geometric ray screen rather than applying the
            # coherent approximation's phase-domain gates to a particle bundle.
            x, tx, y, ty = kick.apply(x, tx, y, ty)
        if saved < ns and j == save_index[saved]:
            X[saved], TX[saved], Y[saved], TY[saved] = x, tx, y, ty
            if time is not None:
                T[saved] = time
            saved += 1
        if captured < nc and j == checkpoint_index[captured]:
            CX[captured], CTX[captured], CY[captured], CTY[captured] = x, tx, y, ty
            if time is not None:
                CT[captured] = time
            captured += 1
        if j == step_m.size:
            continue
        a, b, c = 2*j, 2*j+1, 2*j+2
        d = 3*j
        before = (x.copy(), tx.copy(), y.copy(), ty.copy()) if step_operator is not None else None
        g = larmor_values[[a,b,c], None] * inverse_momentum[None, :]
        active = tuple(item for item, lower, upper in supports
                       if lower < z_mm[j+1] and upper > z_mm[j])
        if not active:
            step = canonical_rk4_step_with_time if time is not None else canonical_rk4_step
            result = step(
                x, tx, y, ty, step_m[j], *g,
                kx[a], kx[b], kx[c], ky[a], ky[b], ky[c],
                hn[a], hn[b], hn[c], hs[a], hs[b], hs[c],
                kxy[a], kxy[b], kxy[c],
                *((inverse_speed,) if time is not None else ()),
                magnetic_scale,
                -charge_over_p*dipole_by_t[d], -charge_over_p*dipole_by_t[d+1], -charge_over_p*dipole_by_t[d+2],
                charge_over_p*dipole_bx_t[d], charge_over_p*dipole_bx_t[d+1], charge_over_p*dipole_bx_t[d+2],
            )
            x, tx, y, ty = result[:4]
            if time is not None:
                time += result[4]
            if step_operator is not None:
                x, tx, y, ty = step_operator(j, before, (x, tx, y, ty))
            continue

        def derivative(values, stage, axial_mm):
            xx, px, yy, py = values
            gg = g[stage]
            ux, uy = px + gg*yy, py - gg*xx
            pos = xp.column_stack((xx, yy, xp.full(nr, axial_mm*1e-3)))
            if defer_nonfinite_until_clipping:
                # Preserve NaN placeholders for divergent histories until the
                # caller applies physical stops. Never submit them as XYZ map
                # queries; a non-finite *pre-stop* state must still be rejected.
                valid = xp.all(xp.isfinite(values), axis=0)
                acceleration = xp.full((nr, 2), np.nan)
                if xp.any(valid):
                    acceleration[valid] = magnetic_slope_derivative(active, pos[valid],
                        xp.column_stack((ux[valid], uy[valid])), charge_over_p[valid], xp=xp)
            else:
                acceleration = magnetic_slope_derivative(active, pos,
                    xp.column_stack((ux, uy)), charge_over_p, xp=xp)
            fx, fy = acceleration.T
            fx -= charge_over_p * dipole_by_t[d+stage]
            fy += charge_over_p * dipole_bx_t[d+stage]
            index = (a,b,c)[stage]
            hu, hv = xx*xx - yy*yy, 2.0*xx*yy
            return xp.stack((ux,
                -(kx[index]*magnetic_scale+gg*gg)*xx - kxy[index]*magnetic_scale*yy + gg*py - magnetic_scale*(hn[index]*hu + hs[index]*hv) + fx,
                uy,
                -(ky[index]*magnetic_scale+gg*gg)*yy - kxy[index]*magnetic_scale*xx - gg*px + magnetic_scale*(hn[index]*hv - hs[index]*hu) + fy))

        initial = xp.stack((x, tx-g[0]*y, y, ty+g[0]*x))
        h = float(step_m[j])
        # A map is zero outside its volume. Sample boundary nodes from the
        # interval interior to avoid giving a discontinuous edge finite weight
        # in the neighbouring vacuum interval.
        za = np.nextafter(float(z_mm[j]), float(z_mm[j+1]))
        zc = np.nextafter(float(z_mm[j+1]), float(z_mm[j]))
        zm = 0.5*(float(z_mm[j])+float(z_mm[j+1]))
        k1 = derivative(initial, 0, za)
        k2 = derivative(initial+0.5*h*k1, 1, zm)
        k3 = derivative(initial+0.5*h*k2, 1, zm)
        k4 = derivative(initial+h*k3, 2, zc)
        if time is not None:
            time += h*inverse_speed*(
                xp.sqrt(1.+k1[0]**2+k1[2]**2)
                + 2.*xp.sqrt(1.+k2[0]**2+k2[2]**2)
                + 2.*xp.sqrt(1.+k3[0]**2+k3[2]**2)
                + xp.sqrt(1.+k4[0]**2+k4[2]**2))/6.
        xx, px, yy, py = initial + h*(k1+2*k2+2*k3+k4)/6.0
        x, tx, y, ty = xx, px+g[2]*yy, yy, py-g[2]*xx
        if step_operator is not None:
            x, tx, y, ty = step_operator(j, before, (x, tx, y, ty))
        invalid = ~xp.all(xp.isfinite(xp.stack((x, tx, y, ty))), axis=0) | (xp.hypot(tx, ty) > 100.0)
        if xp.any(invalid):
            if not defer_nonfinite_until_clipping:
                raise ValueError("Vector field trajectory leaves the forward-Z domain; reduce the step or use time-domain transport")
            x[invalid] = tx[invalid] = y[invalid] = ty[invalid] = np.nan
            if time is not None:
                time[invalid] = np.nan
    result = X, TX, Y, TY, CX, CTX, CY, CTY
    return (*result, T, CT) if time is not None else result

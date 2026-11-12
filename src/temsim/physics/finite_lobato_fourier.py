"""Finite-depth Lobato potentials from continuous Fourier integrals.

The nuclear core is integrated before sampling. No sampling-dependent radial
core clamp, infinite-depth projection, or specimen pixel mask is used. The
finite-CIF fast path requires a proved quarter-period transverse lattice;
actual finite atomic sites are selected before its structure factors are made.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np


def finite_lobato_fourier_coefficients(q_squared, z_lower, z_upper,
                                      scaled_a, scaled_mu, *, xp=np):
    """Return integral(dz dx dy V exp(-2 pi i q.r)), in V Angstrom^3.

    Frequencies are cycles/Angstrom; bounds are Angstrom relative to one atom.
    ``scaled_a, scaled_mu`` are the unmodified abTEM Lobato potential parameters
    for V(r) = sum(a * (2/(mu*r)+1) * exp(-mu*r)). Inputs broadcast normally.
    """
    q2, lo, hi = xp.broadcast_arrays(xp.asarray(q_squared, dtype=xp.float64),
        xp.asarray(z_lower, dtype=xp.float64), xp.asarray(z_upper, dtype=xp.float64))
    a, mu = np.asarray(scaled_a, dtype=float), np.asarray(scaled_mu, dtype=float)
    if (a.ndim != 1 or a.shape != mu.shape or not len(a)
            or not np.isfinite(a).all() or not np.isfinite(mu).all() or np.any(mu <= 0)):
        raise ValueError("Lobato coefficients require finite vectors and positive inverse lengths")
    if (not bool(xp.all(xp.isfinite(q2))) or bool(xp.any(q2 < 0))
            or not bool(xp.all(xp.isfinite(lo))) or not bool(xp.all(xp.isfinite(hi)))
            or bool(xp.any(hi < lo))):
        raise ValueError("Finite Fourier frequencies and ordered bounds must be finite")
    result = xp.zeros(q2.shape, dtype=xp.float64)
    left, right = xp.minimum(abs(lo), abs(hi)), xp.maximum(abs(lo), abs(hi))
    crossing = (lo < 0) & (hi > 0)
    for amplitude, inverse_length in zip(a, mu):
        beta = xp.sqrt(inverse_length**2+4*np.pi**2*q2)
        c1, c2 = 2/(inverse_length*beta**2), inverse_length/beta**4

        def positive(lower, upper):
            u, d = beta*lower, beta*(upper-lower)
            return (2*np.pi*amplitude*xp.exp(-u)
                *((c1+c2*(u+2))*(-xp.expm1(-d))-c2*d*xp.exp(-d)))

        direct = positive(left, right)
        split = positive(xp.zeros_like(lo), abs(lo))+positive(xp.zeros_like(hi), abs(hi))
        result += xp.where(crossing, split, direct)
    return result


def lobato_xy_fourier(q_squared, z, scaled_a, scaled_mu, *, xp=np,
                     absolute_parameters=False):
    """Continuous transverse Fourier transform at one depth, V Angstrom^2."""
    q2, z = xp.broadcast_arrays(xp.asarray(q_squared, dtype=xp.float64),
                                abs(xp.asarray(z, dtype=xp.float64)))
    result = xp.zeros(q2.shape, dtype=xp.float64)
    a = abs(np.asarray(scaled_a)) if absolute_parameters else np.asarray(scaled_a)
    for amplitude, inverse_length in zip(a, np.asarray(scaled_mu)):
        beta = xp.sqrt(inverse_length**2+4*np.pi**2*q2)
        result += (2*np.pi*amplitude*xp.exp(-beta*z)
            *(2/(inverse_length*beta)+inverse_length*(z/beta**2+1/beta**3)))
    return result


def _host(value, xp):
    return np.asarray(value) if xp is np else xp.asnumpy(value)


@lru_cache(maxsize=128)
def _parameters(number):
    from ase.data import chemical_symbols
    from abtem.parametrizations import LobatoParametrization
    return LobatoParametrization().scaled_parameters(chemical_symbols[number], "potential").astype(np.float64)


@dataclass(frozen=True)
class _Group:
    number: int
    z: np.ndarray
    structure: object
    atoms_per_plane: int


def _chain(q2, lower, upper, z, a, mu, *, xp):
    """Sum one finite arithmetic atomic chain without an infinite-z image."""
    z = np.sort(np.asarray(z, dtype=float))
    if not len(z):
        return xp.zeros_like(q2)
    if len(z) == 1:
        return finite_lobato_fourier_coefficients(q2, lower-z[0], upper-z[0], a, mu, xp=xp)
    step = (z[-1]-z[0])/(len(z)-1)
    ideal = z[0]+np.arange(len(z))*step
    residual = float(np.max(abs(z-ideal)))
    if step <= 0 or residual > 5e-8:
        # General finite chains remain exact, with no nearest-plane deposition.
        result = xp.zeros_like(q2)
        for value in z:
            result += finite_lobato_fourier_coefficients(q2, lower-value, upper-value, a, mu, xp=xp)
        return result
    left = int(np.searchsorted(ideal, lower, side="left"))
    right = int(np.searchsorted(ideal, upper, side="right"))
    output = xp.zeros_like(q2)
    for amplitude, inverse_length in zip(a, mu):
        beta = xp.sqrt(inverse_length**2+4*np.pi**2*q2)
        c1, c2 = 2/(inverse_length*beta**2), inverse_length/beta**4
        d = beta*(upper-lower)
        t, decay = -xp.expm1(-d), xp.exp(-d)
        h = beta*step
        denominator = -xp.expm1(-h)

        def outside(distance, population):
            if population <= 0:
                return xp.zeros_like(q2)
            r = xp.exp(-h)
            power = xp.exp(-h*population)
            sum0 = -xp.expm1(-h*population)/denominator
            sum1 = (r*sum0-population*power)/denominator
            if population == 1:
                sum1 = xp.zeros_like(q2)
            moment0 = xp.exp(-beta*distance)*sum0
            moment1 = xp.exp(-beta*distance)*(distance*sum0+step*sum1)
            return 2*np.pi*amplitude*((c1+2*c2)*t*moment0
                +c2*beta*t*moment1-c2*d*decay*moment0)

        output += outside(lower-ideal[left-1], left) if left else 0
        output += outside(ideal[right]-upper, len(z)-right) if right < len(z) else 0
    for value in ideal[left:right]:
        output += finite_lobato_fourier_coefficients(q2, lower-value, upper-value, a, mu, xp=xp)
    return output


def _real_projection(coefficients, *, xp):
    """Real Fourier Galerkin projection, including Nyquist hyperplanes."""
    ny, nx = coefficients.shape
    partner = coefficients[(-xp.arange(ny)) % ny][:, (-xp.arange(nx)) % nx].conj()
    return (coefficients+partner)/2


def _registration_bound(groups, q2, area, xy_residual, z_residual, *, xp):
    """All-slice Fourier L1 bound for admitted geometric roundoff only."""
    result = xp.asarray(0., dtype=xp.float64)
    q = xp.sqrt(q2)
    for group in groups:
        a, mu = _parameters(group.number)
        infinite = xp.zeros_like(q2)
        for amplitude, inverse_length in zip(abs(a), mu):
            beta2 = inverse_length**2+4*np.pi**2*q2
            infinite += 8*np.pi*amplitude*(1/(inverse_length*beta2)+inverse_length/beta2**2)
        xy = 2*np.pi*q*xy_residual*infinite
        zz = 2*z_residual*lobato_xy_fourier(q2, 0., abs(a), mu, xp=xp)
        result += len(group.z)*group.atoms_per_plane*xp.sum(xy+zz)/area
    return float(_host(result, xp))


def _build(groups, extent, shape, count, length, *, xp, output,
           cancelled, progress_callback=None):
    ny, nx = map(int, shape)
    fx, fy = xp.meshgrid(xp.fft.fftfreq(nx, extent[0]/nx), xp.fft.fftfreq(ny, extent[1]/ny))
    q2 = fx*fx+fy*fy
    area = float(np.prod(extent))
    edges = np.linspace(0., length, count+1, dtype=np.float64)
    rounding_sum, largest_imaginary = 0., 0.
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if cancelled():
            raise InterruptedError("Finite Lobato Fourier material preparation cancelled")
        coefficients = xp.zeros((ny, nx), dtype=xp.complex128)
        for group in groups:
            a, mu = _parameters(group.number)
            coefficients += group.structure*_chain(q2, lower, upper, group.z, a, mu, xp=xp)
        coefficients = _real_projection(coefficients, xp=xp)
        values = xp.fft.ifft2(coefficients*(nx*ny/area))
        largest_imaginary = max(largest_imaginary, float(_host(xp.max(abs(values.imag)), xp)))
        if largest_imaginary > 1e-9:
            raise ValueError("Finite Fourier potential failed its Hermitian reality check")
        real = values.real
        stored = real.astype(xp.float32)
        rounding_sum += float(_host(xp.max(abs(real-stored.astype(xp.float64))), xp))
        output(index, _host(stored, xp))
        if progress_callback is not None and (index+1 == count or (index+1) % max(1, count//20) == 0):
            progress_callback(index+1, count, "Analytic finite Lobato potential slices")
    return np.diff(edges), rounding_sum, largest_imaginary, q2


def _error_record(rounding, registration, images, energy_kev):
    from temsim.physics.wave_imaging import interaction_constant_rad_per_v_angstrom
    sigma = interaction_constant_rad_per_v_angstrom(energy_kev)
    phases = [sigma*rounding, sigma*registration, sigma*images]
    total = sum(phases)
    return {"rounding_phase_bound_rad": phases[0], "registration_phase_bound_rad": phases[1],
        "periodic_image_phase_bound_rad": phases[2], "total_phase_bound_rad": total,
        "detector_absolute_error_bound_per_incident_weight": 2*total+total*total,
        "scope": "Potential storage, geometric registration and padded XY images; additional discretization is qualified by the unchanged scan comparisons"}


def _effective_basis(crystal):
    repeat, basis, _, proof = crystal
    common = np.asarray(proof.get("basis_common_offset_angstrom_xyz", (0., 0., 0.)), dtype=float)
    if common.shape != (3,) or not np.isfinite(common).all():
        raise ValueError("Crystal proof has an invalid measured common coordinate offset")
    if np.any(common):
        return np.remainder(np.asarray(basis)+common, repeat)
    return np.asarray(basis)


def validate_periodic_atomic_faces(scene, crystal):
    """Reject ambiguous finite atomic membership before source/GPU allocation."""
    from temsim.specimen.lattice_reference import lattice_phase_for_sample
    repeat, _, _, proof = crystal
    basis = _effective_basis(crystal)
    length = scene.thickness_nm*10
    origin = lattice_phase_for_sample(scene)["origin_depth_below_upper_nm"]*10
    bound = np.asarray(proof.get("basis_rounding_error_bound_angstrom_xyz",
                                5.5001e-10*np.asarray(repeat)), dtype=float)
    if bound.shape != (3,) or not np.isfinite(bound).all() or np.any(bound < 0):
        raise ValueError("Crystal proof has an invalid basis rounding certificate")
    minimum, largest_roundoff = math.inf, 0.
    for atom in basis:
        first = math.floor((-origin-atom[2])/repeat[2])-1
        last = math.ceil((length-origin-atom[2])/repeat[2])+1
        z = origin+atom[2]+np.arange(first, last+1)*repeat[2]
        distance = float(np.min(np.minimum(abs(z), abs(z-length))))
        roundoff = 4*float(np.spacing(max(1., abs(origin), length, abs(atom[2]), abs(repeat[2]))))
        minimum, largest_roundoff = min(minimum, distance), max(largest_roundoff, roundoff)
        if distance <= bound[2]+roundoff:
            raise ValueError("Rectangular basis rounding makes finite-slab atomic face membership ambiguous")
    return {"rectangular_basis_rounding_error_bound_angstrom_xyz": bound.tolist(),
        "minimum_atomic_distance_to_slab_faces_angstrom": minimum,
        "maximum_atomic_face_floating_tolerance_angstrom": largest_roundoff,
        "atomic_face_membership_guard": "Reject when basis rounding plus four floating-point ULPs can alter finite-slab inclusion"}


def build_periodic_lobato_potential(crystal, scene, fov_xy, tiles_xy, pixels, count,
        *, energy_kev=300., cancelled=lambda: False, xp=np, progress_callback=None,
        maximum_device_working_bytes=24*1024**3):
    """Return only the real float32 primitive, actual slices and provenance."""
    from temsim.specimen.lattice_reference import lattice_phase_for_sample
    from temsim.physics.host_memory_budget import check_host_allocation
    repeat, _, numbers, proof = crystal
    basis = _effective_basis(crystal)
    face_proof = validate_periodic_atomic_faces(scene, crystal)
    tx, ty = tiles_xy
    pixels = math.ceil(pixels/math.lcm(tx, ty))*math.lcm(tx, ty)
    nx, ny = pixels//tx, pixels//ty
    length = scene.thickness_nm*10
    basis_bound = np.asarray(face_proof["rectangular_basis_rounding_error_bound_angstrom_xyz"])
    scratch = (512+16*len(basis))*ny*nx
    check_host_allocation(4*count*ny*nx+(scratch if xp is np else 8*ny*nx))
    if xp is not np:
        free, _ = xp.cuda.runtime.memGetInfo()
        if scratch > maximum_device_working_bytes or scratch > .8*free:
            raise MemoryError("Analytic primitive slice scratch exceeds its admitted CUDA workspace")
    phase = lattice_phase_for_sample(scene)
    origin = phase["origin_depth_below_upper_nm"]*10
    x, y = xp.meshgrid(xp.fft.fftfreq(nx, repeat[0]/nx), xp.fft.fftfreq(ny, repeat[1]/ny))
    grouped = {}
    plane_residual = 0.
    shift = np.asarray(scene.centre_xy_nm)*10+np.asarray(fov_xy)/2
    for atom, number in zip(basis, numbers):
        first = math.floor((-origin-atom[2])/repeat[2])-1
        last = math.ceil((length-origin-atom[2])/repeat[2])+1
        z = origin+atom[2]+np.arange(first, last+1)*repeat[2]
        z = z[(z >= 0) & (z < length)]
        structure = xp.exp(-2j*np.pi*(x*(atom[0]+shift[0])+y*(atom[1]+shift[1])))
        key = (int(number), tuple(np.rint(z/1e-7).astype(np.int64)))
        if key in grouped:
            previous = grouped[key]
            plane_residual = max(plane_residual, float(np.max(abs(previous.z-z))))
            grouped[key] = _Group(previous.number, previous.z,
                previous.structure+structure, previous.atoms_per_plane+1)
        else:
            grouped[key] = _Group(int(number), z, structure, 1)
    groups = list(grouped.values())
    result = np.empty((count, ny, nx), dtype=np.float32)
    slices, rounding, imaginary, q2 = _build(groups, repeat[:2], (ny, nx), count, length,
        xp=xp, output=lambda i, value: result.__setitem__(i, value),
        cancelled=cancelled, progress_callback=progress_callback)
    residual = max((float(np.max(abs(g.z-(g.z[0]+np.arange(len(g.z))*(g.z[-1]-g.z[0])/(len(g.z)-1)))))
                    for g in groups if len(g.z) > 1), default=0.)
    residual += plane_residual
    # The rectangular proof rounds fractional coordinates first to ten and
    # then nine decimal places. Charge that pre-existing basis approximation
    # as well as the arithmetic-chain/plane regrouping; never call it exact
    # registration to the original, unrounded CIF sites.
    registration = _registration_bound(groups, q2, np.prod(repeat[:2]),
        float(np.linalg.norm(basis_bound[:2])), residual+float(basis_bound[2]), xp=xp)
    return result, slices, {**proof, **face_proof, "grid_yx": [pixels, pixels], "primitive_grid_yx": [ny, nx],
        "slices": count, "slice_thicknesses_angstrom": slices.tolist(),
        "supercell_angstrom_xy": list(fov_xy), "tiles_xy": list(tiles_xy),
        "potential_method": "Continuous analytic finite-z Lobato Fourier projection",
        "potential_storage_dtype": "float32", "coefficient_precision": "float64 / complex128",
        "nuclear_core_clamp": None, "infinite_z_projection": False,
        "real_fourier_projection": "Hermitian average, including even-grid Nyquist hyperplanes",
        "maximum_ifft_imaginary_v_angstrom": imaginary, "lattice_phase": phase,
        "maximum_chain_registration_residual_angstrom": residual,
        "atomic_planar_pattern_count": len(groups),
        "common_basis_offset_restored": bool(np.any(proof.get("basis_common_offset_angstrom_xyz", (0., 0., 0.)))),
        "effective_basis_scope": "Original stored basis plus measured common CIF rounding offset, wrapped by the proven repeat; source CIF and stored basis arrays are unchanged",
        "estimated_builder_slice_scratch_bytes": scratch,
        "finite_axial_padding_angstrom": 0.,
        "axial_boundary": "Only actual atoms in the original finite slab; finite slice bounds integrated analytically",
        "numerical_potential_error": _error_record(rounding, registration, 0., energy_kev)}


def periodic_image_potential_bound(numbers, scaled_parameters, padding_angstrom,
                                   minimum_cell_length_angstrom):
    """Uniform all-XY-image potential bound, including every Chebyshev shell."""
    p, length = float(padding_angstrom), float(minimum_cell_length_angstrom)
    if not np.isfinite(p) or p <= 0 or not np.isfinite(length) or length <= 0:
        raise ValueError("Finite image certificate requires positive padding and cell lengths")
    result = 0.
    for number, population in zip(*np.unique(numbers, return_counts=True)):
        a, mu = scaled_parameters[int(number)]
        result += population*float(np.sum(8*abs(a)*(1+2/(mu*p))*np.exp(-mu*p)
                                          /(-np.expm1(-mu*length))**2))
    return result


def build_finite_lobato_potential(scene, crystal, fov_xy, shape_yx, count, *,
        maximum_working_bytes, energy_kev=300., cancelled=lambda: False,
        xp=np, progress_callback=None, maximum_device_working_bytes=24*1024**3):
    """Finite quarter-lattice CIF atoms, padded exact structure factors, ROI.

    All physical atoms are retained. The finite specimen is never synthesized
    from an infinite periodic unit cell or multiplied by a potential mask.
    """
    from temsim.specimen.atomistic import build_cif_equilibrium_atoms
    from temsim.specimen.geometry import quaternion_to_matrix
    from temsim.specimen.lattice_reference import lattice_phase_for_sample, lattice_translation_z_angstrom
    from temsim.physics.host_memory_budget import check_host_allocation
    repeat = np.asarray(crystal[0])
    ny, nx = map(int, shape_yx)
    extent = np.asarray(fov_xy, dtype=float)
    tiles = np.rint(extent/repeat[:2]).astype(int)
    if np.max(abs(extent-tiles*repeat[:2])) > 5e-8 or nx % tiles[0] or ny % tiles[1]:
        raise ValueError("Finite analytic reference requires a commensurate physical crystal domain")
    primitive_pixels = np.array([nx//tiles[0], ny//tiles[1]])
    # Include the entire physical cylinder, as well as twenty Angstrom between
    # the observed ROI and the padded Fourier box. No external atomic tail is
    # lost by an intermediate ROI atom crop.
    physical = np.asarray(scene.size_xy_nm)*10+2*abs(np.asarray(scene.centre_xy_nm)*10)
    # The image remainder is bounded on the *entire* Fourier cell before its
    # Galerkin projection. Therefore atoms, as well as the observation ROI,
    # need twenty Angstrom of vacuum to every cell edge.
    minimum = np.maximum(extent+40, physical+40)
    padded_tiles = np.ceil(minimum/repeat[:2]/2).astype(int)*2
    padded_extent = padded_tiles*repeat[:2]
    padded_nx, padded_ny = primitive_pixels*padded_tiles
    px, py = (np.array([padded_nx, padded_ny])-np.array([nx, ny]))//2
    if (padded_nx-nx) % 2 or (padded_ny-ny) % 2:
        raise ValueError("Finite Fourier crop is not centre aligned")
    output_bytes = 4*count*nx*ny
    # The pattern count is known before any dense Fourier lattice allocation.
    # The exact whole-stage admission is performed below after the small atom
    # selection, and before allocating the q-grid or any structure factors.
    length = scene.thickness_nm*10
    atoms, _ = build_cif_equilibrium_atoms(scene.cif_path, thickness_angstrom=length,
        field_of_view_angstrom=tuple(padded_extent),
        rotation_matrix=quaternion_to_matrix(scene.orientation_quaternion_wxyz),
        specimen_size_xy_angstrom=tuple(np.asarray(scene.size_xy_nm)*10),
        specimen_envelope_shape=scene.envelope_shape,
        specimen_centre_xy_angstrom=tuple(np.asarray(scene.centre_xy_nm)*10),
        lattice_translation_z_angstrom=lattice_translation_z_angstrom(scene))
    positions = np.asarray(atoms.positions, dtype=float).copy()
    positions[:, :2] -= padded_extent/2
    # Finite builder retains its existing face tolerance. Preserve every
    # selected atomic site, including a possible exactly-on-face atom.
    origin = np.asarray(scene.centre_xy_nm)*10
    pitch = repeat[:2]/4
    indices = np.rint((positions[:, :2]-origin)/pitch).astype(np.int64)
    xy_residual = float(np.max(np.linalg.norm(positions[:, :2]-(indices*pitch+origin), axis=1)))
    if xy_residual > 5e-8:
        raise ValueError("Finite CIF atoms do not meet the explicit quarter-lattice Fourier qualification")
    planes = {}
    for index, (number, xyz) in enumerate(zip(atoms.numbers, positions)):
        planes.setdefault((int(number), int(np.rint(xyz[2]/1e-8))), []).append(index)
    patterns, plane_residual, chain_residual = {}, 0., 0.
    for (number, _), selection in planes.items():
        selection = np.asarray(selection)
        xy = indices[selection]
        xy = xy[np.lexsort((xy[:, 0], xy[:, 1]))]
        key = (number, xy.tobytes())
        depth = float(np.mean(positions[selection, 2]))
        plane_residual = max(plane_residual, float(np.max(abs(positions[selection, 2]-depth))))
        if key not in patterns:
            patterns[key] = (xy, [])
        patterns[key][1].append(depth)
    scratch_bytes = (512+16*len(patterns))*int(padded_nx)*int(padded_ny)
    check_host_allocation(output_bytes+(scratch_bytes if xp is np else 8*int(padded_nx)*int(padded_ny)),
                          requested_maximum_bytes=maximum_working_bytes)
    if xp is not np:
        free, _ = xp.cuda.runtime.memGetInfo()
        if scratch_bytes > .8*free or scratch_bytes > maximum_device_working_bytes:
            raise MemoryError("Analytic finite-CIF slice scratch exceeds the bounded CUDA workspace")
    fx, fy = xp.meshgrid(xp.fft.fftfreq(int(padded_nx), padded_extent[0]/padded_nx),
                         xp.fft.fftfreq(int(padded_ny), padded_extent[1]/padded_ny))
    kx = xp.fft.fftfreq(int(padded_nx), 1/int(padded_nx)).round().astype(xp.int64)
    ky = xp.fft.fftfreq(int(padded_ny), 1/int(padded_ny)).round().astype(xp.int64)
    phase = xp.exp(-2j*np.pi*(fx*(origin[0]+padded_extent[0]/2)+fy*(origin[1]+padded_extent[1]/2)))
    groups = []
    for (number, _), (xy, depths) in patterns.items():
        small_nx, small_ny = (4*padded_tiles).tolist()
        counts = np.zeros((small_ny, small_nx), dtype=np.float64)
        np.add.at(counts, (xy[:, 1] % small_ny, xy[:, 0] % small_nx), 1.)
        transform = xp.fft.fft2(xp.asarray(counts))
        structure = transform[ky % small_ny][:, kx % small_nx]*phase
        zz = np.sort(depths)
        if len(zz) > 1:
            residual = float(np.max(abs(zz-(zz[0]+np.arange(len(zz))*(zz[-1]-zz[0])/(len(zz)-1)))))
            if residual <= 5e-8:
                chain_residual = max(chain_residual, residual)
        groups.append(_Group(number, zz, structure, len(xy)))
    # Structure factors, q-lattice and slice scratch are device/host resident,
    # while the only depth-sized output is the requested float32 ROI.
    del fx, fy, phase
    result = np.empty((count, ny, nx), dtype=np.float32)
    slices, rounding, imaginary, q2 = _build(groups, padded_extent,
        (int(padded_ny), int(padded_nx)), count, length, xp=xp,
        output=lambda i, value: result.__setitem__(i, value[py:py+ny, px:px+nx]),
        cancelled=cancelled, progress_callback=progress_callback)
    z_residual = plane_residual+chain_residual
    registration = _registration_bound(groups, q2, np.prod(padded_extent), xy_residual, z_residual, xp=xp)
    roi_padding = float(np.min((padded_extent-extent)/2))
    escape = float(np.max(np.maximum(abs(positions[:, :2])-padded_extent/2, 0.)))
    pad_effective = float(np.min(padded_extent/2-abs(positions[:, :2])))
    image_potential = periodic_image_potential_bound(atoms.numbers,
        {g.number: _parameters(g.number) for g in groups}, pad_effective, float(np.min(padded_extent)))
    # Each continuous Fourier coefficient is bounded by the full-cell
    # supremum. Summing the represented rectangle gives this deliberately
    # conservative L-infinity stability bound, also covering the Nyquist
    # Hermitian average. No L-infinity contraction is assumed for truncation.
    fourier_stability = int(padded_nx)*int(padded_ny)
    return result, slices, {"model": "Actual finite CIF atoms; analytic finite-z Lobato Fourier projection with certified XY image remainder",
        "atom_count": len(atoms), "atomic_planar_pattern_count": len(groups),
        "grid_yx": [ny, nx], "slices": count, "supercell_angstrom_xy": extent.tolist(),
        "padded_grid_yx": [int(padded_ny), int(padded_nx)],
        "padded_fov_angstrom_xy": padded_extent.tolist(), "padded_tiles_xy": padded_tiles.tolist(),
        "whole_physical_specimen_selected": True, "atomic_extents_angstrom_xyz":
            [positions.min(axis=0).tolist(), positions.max(axis=0).tolist()],
        "padding_effective_angstrom": pad_effective, "maximum_atom_escape_angstrom": escape,
        "roi_padding_angstrom": roi_padding,
        "continuous_all_images_potential_sup_bound_v": image_potential,
        "fourier_projection_linf_stability_bound": fourier_stability,
        "image_certificate_scope": "Full padded-cell continuous image supremum times represented Fourier rank, then actual slab length and interaction constant",
        "maximum_xy_registration_residual_angstrom": xy_residual,
        "maximum_z_registration_residual_angstrom": z_residual,
        "potential_storage_dtype": "float32", "coefficient_precision": "float64 / complex128",
        "nuclear_core_clamp": None, "infinite_z_projection": False,
        "physical_atomic_site_selection_before_potential": True,
        "potential_pixel_envelope_mask_applied": False,
        "maximum_ifft_imaginary_v_angstrom": imaginary,
        "lattice_phase": lattice_phase_for_sample(scene),
        "estimated_host_output_bytes": output_bytes, "estimated_slice_scratch_bytes": scratch_bytes,
        "slice_thicknesses_angstrom": slices.tolist(),
        "numerical_potential_error": _error_record(rounding, registration,
            image_potential*length*fourier_stability, energy_kev)}


__all__ = ["finite_lobato_fourier_coefficients", "lobato_xy_fourier",
           "build_periodic_lobato_potential", "build_finite_lobato_potential",
           "periodic_image_potential_bound", "validate_periodic_atomic_faces"]

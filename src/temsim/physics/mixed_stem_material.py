"""Finite CIF reference potentials without a pixel mask of atomic tails."""
import math

import numpy as np


def validate_analytic_crystal(scene,crystal):
    """Prove the finite-reference XY indexing before particle/wave preparation.

    The Fourier finite reference currently groups the actual selected sites
    on a rectangular, quarter-cell XY lattice. Unsupported CIFs fail clearly;
    they are never sent to the former radial-core-clamped material builder.
    """
    if crystal is None:
        raise ValueError("Mixed STEM analytic material requires a proven rectangular CIF crystal with quarter-lattice XY sites")
    repeat,basis,numbers,_=crystal
    repeat=np.asarray(repeat,dtype=float)
    basis=np.asarray(basis,dtype=float)
    numbers=np.asarray(numbers)
    if (repeat.shape!=(3,) or not np.isfinite(repeat).all() or np.any(repeat<=0)
            or basis.ndim!=2 or basis.shape[1]!=3 or not len(basis)
            or not np.isfinite(basis).all() or numbers.shape!=(len(basis),)
            or numbers.dtype.kind not in 'iu' or np.any(numbers<=0)):
        raise ValueError("Mixed STEM analytic material has invalid rectangular crystal proof data")
    pitch=repeat[:2]/4
    residual=float(np.max(np.linalg.norm(basis[:,:2]-np.rint(basis[:,:2]/pitch)*pitch,axis=1)))
    if residual>5e-8:
        raise ValueError("Mixed STEM finite Fourier reference requires proven quarter-lattice XY sites; this CIF is unsupported")
    from temsim.physics.finite_lobato_fourier import validate_periodic_atomic_faces
    face_proof=validate_periodic_atomic_faces(scene,crystal)
    return {"scope":"Proven rectangular CIF with quarter-lattice XY indexing of every actual finite-specimen site",
        "maximum_basis_xy_registration_residual_angstrom":residual,
        "quarter_lattice_pitch_angstrom_xy":pitch.tolist(),
        "radial_core_clamped_fallback":False,"finite_slab_membership":face_proof}


def _legacy_abtem_finite_cif_potential(scene, fov_xy_angstrom, shape_yx, count, *,
                         maximum_working_bytes, cancelled=lambda: False):
    """Crop physical atomic sites, include screened-potential padding, then ROI.

    Atomic sites outside the physical specimen are excluded before potential
    integration. Atomic potentials extending across a specimen boundary remain
    present. Lateral/axial numerical padding isolates the actual finite slab;
    no specimen-envelope mask is multiplied into the potential pixels.
    """
    import abtem
    from temsim.physics.host_memory_budget import check_host_allocation
    from temsim.specimen.atomistic import build_cif_equilibrium_atoms
    from temsim.specimen.geometry import quaternion_to_matrix
    from temsim.specimen.lattice_reference import lattice_phase_for_sample, lattice_translation_z_angstrom
    if scene.source_kind != "cif":
        raise ValueError("Finite mixed STEM material reference requires its actual CIF")
    ny, nx = map(int, shape_yx)
    fov_x, fov_y = map(float, fov_xy_angstrom)
    dx, dy = fov_x/nx, fov_y/ny
    length = scene.thickness_nm*10
    dz = length/count
    pad_x, pad_y, pad_z = (math.ceil(20/step) for step in (dx, dy, dz))
    padded_fov = (fov_x+2*pad_x*dx, fov_y+2*pad_y*dy)
    padded_shape = (ny+2*pad_y, nx+2*pad_x)
    # One output, two small builder slabs, and coordinates/atom scratch.
    block = min(8, count)
    estimated = 4*count*ny*nx+24*block*padded_shape[0]*padded_shape[1]
    if estimated > maximum_working_bytes:
        raise MemoryError("Finite material potential and bounded builder slabs exceed their host-memory budget")
    check_host_allocation(estimated, requested_maximum_bytes=maximum_working_bytes)
    atoms, _ = build_cif_equilibrium_atoms(scene.cif_path,
        thickness_angstrom=length, field_of_view_angstrom=padded_fov,
        rotation_matrix=quaternion_to_matrix(scene.orientation_quaternion_wxyz),
        specimen_size_xy_angstrom=tuple(np.asarray(scene.size_xy_nm)*10),
        specimen_envelope_shape=scene.envelope_shape,
        specimen_centre_xy_angstrom=tuple(np.asarray(scene.centre_xy_nm)*10),
        lattice_translation_z_angstrom=lattice_translation_z_angstrom(scene))
    # Only vacuum is added along Z; actual atomic slab termination is retained.
    atoms.positions[:, 2] += pad_z*dz
    atoms.set_cell((*padded_fov, length+2*pad_z*dz), scale_atoms=False)
    atoms.pbc = False
    abtem.config.set({"diagnostics.progress_bar": False})
    potential = abtem.Potential(atoms, gpts=(padded_shape[1], padded_shape[0]),
        slice_thickness=dz, parametrization="lobato", projection="finite",
        periodic=False, device="cpu")
    radial_cutoff = 0.
    for symbol in set(atoms.get_chemical_symbols()):
        radial = np.asarray(potential.integrator.get_integral_table(symbol, potential.sampling).radial_gpts)
        if len(radial) < 2 or not np.all(np.diff(radial) > 0):
            raise ValueError("Finite atomic potential radial interpolation is unresolved")
        radial_cutoff = max(radial_cutoff, float(radial[-1]))
    if radial_cutoff >= min(pad_x*dx, pad_y*dy, pad_z*dz):
        raise ValueError("Finite CIF reference lacks enough screened-potential padding")
    result = np.empty((count, ny, nx), dtype=np.float32)
    thicknesses = []
    for begin in range(0, count, block):
        if cancelled():
            raise InterruptedError("Mixed STEM finite material preparation cancelled")
        end = min(count, begin+block)
        built = potential.build(first_slice=pad_z+begin, last_slice=pad_z+end, lazy=False)
        values = np.asarray(built.array, dtype=np.float32).transpose(0, 2, 1)
        result[begin:end] = values[:, pad_y:pad_y+ny, pad_x:pad_x+nx]
        thicknesses.extend(np.asarray(built.slice_thickness, dtype=float))
        del built, values
    thicknesses = np.asarray(thicknesses)
    if not np.isclose(thicknesses.sum(), length, atol=1e-10, rtol=0) or not np.isfinite(result).all():
        raise ValueError("Finite CIF material slices do not span the actual slab")
    return result, thicknesses, {
        "model": "Finite original CIF atoms with finite Lobato projection, PBC false",
        "atom_count": len(atoms), "grid_yx": [ny, nx], "slices": count,
        "supercell_angstrom_xy": [fov_x, fov_y], "sampling_angstrom_xy": [dx, dy],
        "potential_storage_dtype": "float32", "projection": "finite", "parametrization": "lobato",
        "screened_potential_cutoff_angstrom": radial_cutoff,
        "vacuum_padding_angstrom_xyz": [pad_x*dx, pad_y*dy, pad_z*dz],
        "builder_slice_block": block, "estimated_host_peak_bytes": estimated,
        "physical_envelope_shape": scene.envelope_shape,
        "physical_specimen_size_nm": list(scene.size_xy_nm),
        "physical_atomic_site_selection_before_potential": True,
        "lattice_phase": lattice_phase_for_sample(scene),
        "potential_pixel_envelope_mask_applied": False,
        "slice_thicknesses_angstrom": thicknesses.tolist()}


def finite_cif_potential(scene,fov_xy_angstrom,shape_yx,count,*,maximum_working_bytes,
        crystal=None,energy_kev=None,xp=None,maximum_device_working_bytes=24*1024**3,
        cancelled=lambda:False,progress_callback=None):
    """Dense finite-reference storage from the same continuous analytic kernel.

    Every selected original atom is retained. An unsupported CIF or failed
    analytic admission is rejected, without selecting the old clamped kernel.
    Explicit ``xp=np`` is reserved for static numerical-operator checks.
    """
    from temsim.physics.finite_lobato_fourier import build_finite_lobato_potential
    from temsim.physics.compute_backend import require_coherent_cuda,cupy_module
    if energy_kev is None:
        raise ValueError("Finite material needs the actual current-weighted source energy")
    if xp is None:
        require_coherent_cuda()
        xp=cupy_module()
    if cancelled():
        raise InterruptedError("Finite material preparation cancelled")
    validate_analytic_crystal(scene,crystal)
    return build_finite_lobato_potential(scene,crystal,fov_xy_angstrom,shape_yx,count,
        maximum_working_bytes=maximum_working_bytes,energy_kev=energy_kev,xp=xp,
        maximum_device_working_bytes=maximum_device_working_bytes,
        cancelled=cancelled,progress_callback=progress_callback)

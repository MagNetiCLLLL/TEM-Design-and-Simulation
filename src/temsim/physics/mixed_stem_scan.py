"""CUDA STEM from an executed particle-to-wave boundary, without a Tip wave.

This local scan uses an exact forward angular-spectrum material kinetic
operator. Static near-uniform axial magnetic overlap is included by a
qualified symmetric Larmor/diamagnetic split, not by the general column wave
kernel. Independent resident calculations qualify both the periodic reduction
and the field split before an image can be published.
"""
from dataclasses import asdict, dataclass, field, replace
from copy import copy
import math
from time import perf_counter

import numpy as np
from scipy.constants import c, e, h, m_e
from scipy.ndimage import map_coordinates

from temsim import input_io
from temsim.immutable_json import freeze_json, thaw_json
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.physics.compute_backend import cupy_module, require_coherent_cuda
from temsim.physics.mixed_stem_operators import (ScanPacket, evaluate_periodic,
    periodic_coefficients, resident_observe, rotate_wave)
from temsim.physics.particle_wave_source import (MixedSourceNumerics,
    ParticleWaveSource, build_particle_wave_checkpoint)
from temsim.physics.stem_wave_imaging import AngularDetector
from temsim.physics.wave_checkpoint import WaveCheckpoint


_PAIRS = (((960, 8), (1152, 16)), ((1152, 16), (1344, 32)),
          ((1344, 32), (1536, 64)), ((1536, 64), (1728, 96)),
          ((1728, 96), (1920, 128)))
_VALIDATION_NM = np.array([[-.49, -.57], [-.17, .19], [.01, .01], [.23, -.11],
                         [.59, .37], [0., 0.], [.02, .135], [.14, .25], [.08, .38], [.19, .45]])


@dataclass(frozen=True)
class MixedStemScanRequest:
    source: MixedSourceNumerics = field(default_factory=MixedSourceNumerics)
    pixels: int = 64
    pitch_nm: float = .02
    detectors: tuple = (AngularDetector("bf", 0., 15.), AngularDetector("adf", 50., 150.))
    batch_size: int = 8
    maximum_working_bytes: int = 24*1024**3
    maximum_host_working_bytes: int = 48*1024**3
    material_levels: tuple[tuple[int, int], ...] | None = None
    maximum_modulation_error: float = .05
    bandwidth_fraction: float = 2/3
    save_diffraction: bool = True

    def __post_init__(self):
        # JSON has lists; copy them into immutable numerical input pairs so a
        # saved executed request can be inspected without changing its values.
        if isinstance(self.material_levels, (tuple, list)):
            object.__setattr__(self, "material_levels", tuple(
                tuple(level) if isinstance(level, (tuple, list)) else level
                for level in self.material_levels))

    def validate(self):
        self.source.validate()
        for name in ("pixels", "batch_size", "maximum_working_bytes", "maximum_host_working_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.pixels > 4096:
            raise ValueError("Mixed STEM raster pixels cannot exceed 4096 per side")
        if self.material_levels is not None:
            if not isinstance(self.material_levels, tuple) or len(self.material_levels) < 2:
                raise ValueError("Explicit material levels require at least two immutable (pixels, slices) pairs")
            previous = None
            for level in self.material_levels:
                if (not isinstance(level, tuple) or len(level) != 2
                        or any(type(value) is not int for value in level)
                        or not 32 <= level[0] <= 16384 or not 1 <= level[1] <= 1048576):
                    raise ValueError("Each material level needs integer pixels in [32, 16384] and slices in [1, 1048576]")
                if previous is not None and (level[0] < previous[0] or level[1] < previous[1] or level == previous):
                    raise ValueError("Material levels must refine at least one coordinate without coarsening the other")
                previous = level
        for name in ("pitch_nm", "maximum_modulation_error", "bandwidth_fraction"):
            v = getattr(self, name)
            if isinstance(v, (bool, np.bool_)) or not np.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.save_diffraction) is not bool:
            raise ValueError("save_diffraction must be a Boolean")
        if self.maximum_modulation_error > .05 or self.bandwidth_fraction > 2/3:
            raise ValueError("Existing 5% modulation and 2/3 anti-alias budgets cannot be relaxed")
        detectors = tuple(self.detectors)
        if not detectors or len({d.key for d in detectors}) != len(detectors):
            raise ValueError("Mixed STEM needs unique angular detector keys")
        for detector in detectors:
            detector.validate()
            if detector.key in {"other", "numerical_band_loss", "source_mode_loss",
                                "source_representation_loss", "upstream_physical_loss"}:
                raise ValueError("Angular detector key is reserved for the absolute source/loss ledger")
            if not all(np.isfinite((detector.inner_mrad, detector.outer_mrad))) or detector.outer_mrad >= 500:
                raise ValueError("Mixed STEM detector limits must be finite and below 500 mrad")
        ordered = sorted((d.inner_mrad, d.outer_mrad) for d in detectors)
        if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Mixed STEM diagnostic angular annuli must not overlap")
        return self


@dataclass(frozen=True)
class MixedStemScanResult:
    scan_x_nm: np.ndarray
    scan_y_nm: np.ndarray
    fractions: dict
    current_a: dict
    diffraction_probability: np.ndarray | None
    qx_inv_angstrom: np.ndarray
    qy_inv_angstrom: np.ndarray
    source_checkpoint: WaveCheckpoint
    record: object


class MixedStemQualificationError(ValueError):
    """Failed scientific qualification with a persistable partial receipt."""

    def __init__(self, message, history):
        super().__init__(message)
        self.record = {"schema": "mixed-stem-qualification-failure-v1",
                       "status": "NOT_QUALIFIED", "history": history}


def _positions(request, positions_nm):
    count = request.pixels**2 if positions_nm is None else int(np.prod(np.shape(positions_nm)[:-1]))
    if count*(80+16*(len(request.detectors)+5)) > request.maximum_host_working_bytes:
        raise MemoryError("Requested scan coordinates and result buffers exceed the declared working-memory budget")
    if positions_nm is None:
        axis = (np.arange(request.pixels)-(request.pixels-1)/2)*request.pitch_nm
        x, y = np.meshgrid(axis, axis)
        positions = np.stack((x, y), axis=-1)
    else:
        positions = np.asarray(positions_nm, dtype=float)
    if positions.ndim not in (2, 3) or positions.shape[-1] != 2 or not positions.size or not np.isfinite(positions).all():
        raise ValueError("Mixed STEM positions must be finite (N,2) or (Ny,Nx,2) nanometres")
    return positions.reshape(-1, 2)*10, positions.shape[:-1], positions[..., 0], positions[..., 1]


def _cancelled_check(cancelled):
    if cancelled():
        raise InterruptedError("Mixed STEM scan cancelled; no partial image published")


def _wavelength(energy_ev):
    return h*c/e*1e10/math.sqrt(energy_ev*(energy_ev+2*m_e*c*c/e))


def _rectangular_crystal(scene):
    """Prove a rectangular lattice from actual CIF and captured orientation."""
    from temsim.specimen.cif_io import read_cif_atoms
    from temsim.specimen.geometry import quaternion_to_matrix
    if scene.is_vacuum or scene.source_kind != "cif":
        return None
    atoms = read_cif_atoms(scene.cif_path)
    rotation = quaternion_to_matrix(scene.orientation_quaternion_wxyz)
    lattice = np.asarray(atoms.cell) @ rotation.T
    candidates = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if i or j or k:
                    candidates.append(np.array((i, j, k)) @ lattice)
    vectors = np.asarray(candidates)
    repeat = []
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        keep = (vectors[:, axis] > 1e-8) & (np.max(abs(vectors[:, other]), axis=1) < 1e-8)
        if not np.any(keep):
            return None
        repeat.append(float(vectors[keep, axis].min()))
    repeat = np.array(repeat)
    ratio = np.prod(repeat)/abs(np.linalg.det(lattice))
    if not np.isclose(ratio, round(ratio), rtol=0, atol=1e-8) or round(ratio) < 1:
        return None
    # Enumerate enough original cells to fill the proven rectangular repeat.
    from itertools import product
    sites, numbers, raw_sites = [], [], []
    largest_coordinate = float(np.max(abs(atoms.positions)))
    for shift in product(range(-4, 5), repeat=3):
        xyz = (atoms.positions+np.asarray(shift) @ np.asarray(atoms.cell)) @ rotation.T
        for number, row in zip(atoms.numbers, xyz):
            raw_sites.append(np.remainder(row, repeat))
            largest_coordinate = max(largest_coordinate, float(np.max(abs(row))))
            sites.append(np.remainder(np.round(row/repeat, 10), 1.))
            numbers.append(number)
    combined = np.c_[numbers, np.round(sites, 9)]
    unique = np.unique(combined, axis=0)
    if len(unique) != round(ratio)*len(atoms):
        return None
    basis, raw = unique[:, 1:]*repeat, np.asarray(raw_sites)
    raw_numbers = np.asarray(numbers)
    matches = np.empty(len(raw), dtype=int)
    displacements = np.empty_like(raw)
    for number in np.unique(raw_numbers):
        selected = np.flatnonzero(raw_numbers == number)
        candidates = np.flatnonzero(unique[:, 0] == number)
        difference = (raw[selected, None, :]-basis[candidates][None, :, :]+repeat/2) % repeat-repeat/2
        closest = np.argmin(np.sum(difference*difference, axis=-1), axis=1)
        matches[selected] = candidates[closest]
        displacements[selected] = difference[np.arange(len(selected)), closest]
    # Restore only the measured common numerical offset lost by round10/
    # round9. The stored basis is unchanged; remaining per-site error is
    # certified against the original unrounded CIF, not assumed to be zero.
    common = (displacements.min(axis=0)+displacements.max(axis=0))/2
    effective = np.remainder(basis+common, repeat)
    residual = (raw-effective[matches]+repeat/2) % repeat-repeat/2
    floating_floor = 32*np.finfo(float).eps*max(1., largest_coordinate, float(np.max(repeat)))
    bound = np.max(abs(residual), axis=0)+floating_floor
    return repeat, basis, unique[:, 0].astype(int), {
        "proof": "Integer CIF lattice translations along all laboratory axes; exact tiled potential checked separately",
        "rectangular_repeat_angstrom_xyz": repeat.tolist(), "rectangular_basis_atoms": len(unique),
        "original_cell_atoms": len(atoms), "original_cell_volume_angstrom3": float(atoms.get_volume()),
        "basis_common_offset_angstrom_xyz": common.tolist(),
        "basis_rounding_error_bound_angstrom_xyz": bound.tolist(),
        "basis_original_rounding_displacement_max_angstrom_xyz": np.max(abs(displacements), axis=0).tolist(),
        "basis_registration_floating_floor_angstrom": floating_floor,
        "basis_registration_raw_site_count": len(raw),
        "basis_registration_scope": "Measured periodic signed differences from original unrounded CIF sites; common numerical offset restored by the analytic builder, remaining per-site error charged",
        "orientation_quaternion_wxyz": list(scene.orientation_quaternion_wxyz)}


def _legacy_abtem_periodic_potential(crystal, scene, fov_xy, tiles_xy, pixels, count,*,cancelled=lambda:False):
    """Explicit diagnostic baseline only; never selected after a new-kernel failure."""
    import abtem
    from ase import Atoms
    from temsim.specimen.lattice_reference import lattice_phase_for_sample
    repeat, basis, numbers, proof = crystal
    tx, ty = tiles_xy
    if pixels % math.lcm(tx, ty):
        pixels = math.ceil(pixels/math.lcm(tx, ty))*math.lcm(tx, ty)
    nx, ny = pixels//tx, pixels//ty
    thickness = scene.thickness_nm*10
    dz = thickness/count
    positions, species = [], []
    phase_record = lattice_phase_for_sample(scene)
    origin_depth_angstrom = phase_record["origin_depth_below_upper_nm"]*10
    for atom, number in zip(basis, numbers):
        first = math.floor((-origin_depth_angstrom-atom[2])/repeat[2])-1
        last = math.ceil((thickness-origin_depth_angstrom-atom[2])/repeat[2])+1
        for iz in range(first, last+1):
            z = origin_depth_angstrom+atom[2]+iz*repeat[2]
            if 0 <= z < thickness:
                positions.append((atom[0], atom[1], z))
                species.append(number)
    padding = max(4*count, math.ceil(20/dz))
    positions = np.asarray(positions)
    positions[:, 2] += padding*dz
    atoms = Atoms(numbers=species, positions=positions,
                  cell=(*repeat[:2], (2*padding+count)*dz), pbc=True)
    abtem.config.set({"diagnostics.progress_bar": False})
    preparation = abtem.Potential(atoms, gpts=(nx, ny), slice_thickness=dz,
        parametrization="lobato", projection="finite", periodic=True, device="cpu")
    for symbol in set(atoms.get_chemical_symbols()):
        radial = np.asarray(preparation.integrator.get_integral_table(symbol, preparation.sampling).radial_gpts)
        if len(radial) < 2 or radial[-1] >= padding*dz:
            raise ValueError("Finite atomic potential lacks enough isolated axial padding")
    built = preparation.build(first_slice=padding, last_slice=padding+count, lazy=False)
    primitive = np.asarray(built.array, dtype=np.float32).transpose(0, 2, 1)
    # Coordinate origin is global zero: the sample translation is a Fourier
    # shift of this periodic potential, not a redefinition of probe phase.
    from temsim.physics.material_slice_storage import compact_legacy_periodic_potential
    result = compact_legacy_periodic_potential(primitive,(tx,ty),fov_xy=fov_xy,
        centre_xy=np.asarray(scene.centre_xy_nm)*10,cancelled=cancelled)
    slices = np.asarray(built.slice_thickness, dtype=float)
    if not np.isclose(slices.sum(), thickness, rtol=0, atol=1e-10):
        raise ValueError("Invalid finite atomic potential or slab termination")
    return result, slices, {**proof, "grid_yx": [pixels, pixels], "slices": count,
        "slice_thicknesses_angstrom": slices.tolist(), "supercell_angstrom_xy": list(fov_xy),
        "tiles_xy": list(tiles_xy), "finite_axial_padding_angstrom": padding*dz,
        "lattice_phase": phase_record,"material_storage":result.literal_periodicity_record(tiles_xy)}


def _periodic_potential(crystal,scene,fov_xy,tiles_xy,pixels,count,*,
        energy_kev=None,xp=None,maximum_device_working_bytes=24*1024**3,
        cancelled=lambda:False,progress_callback=None):
    """Actual finite-z material on CUDA, stored only as its primitive period."""
    from temsim.physics.finite_lobato_fourier import build_periodic_lobato_potential
    from temsim.physics.material_slice_storage import CompactPeriodicPotential
    from temsim.physics.mixed_stem_material import validate_analytic_crystal
    if energy_kev is None:
        raise ValueError("Periodic material needs the actual current-weighted source energy")
    if xp is None:
        require_coherent_cuda()
        xp=cupy_module()
    _cancelled_check(cancelled)
    validate_analytic_crystal(scene,crystal)
    primitive,slices,record=build_periodic_lobato_potential(crystal,scene,fov_xy,tiles_xy,pixels,count,
        energy_kev=energy_kev,xp=xp,maximum_device_working_bytes=maximum_device_working_bytes,
        cancelled=cancelled,progress_callback=progress_callback)
    result=CompactPeriodicPotential(primitive,tiles_xy)
    return result,slices,{**record,"material_storage":result.literal_periodicity_record(tiles_xy)}


@input_io.using_state_inputs
def preflight_mixed_stem_material(state):
    """Read a small CIF proof before costly particle transport or source lift."""
    from temsim.specimen.scene import SpecimenScene
    from temsim.physics.mixed_stem_material import validate_analytic_crystal
    scene=SpecimenScene.from_state(state)
    if scene.is_vacuum or not scene.structure_available:
        raise ValueError("Mixed STEM acquisition requires an actual inserted material structure")
    crystal=_rectangular_crystal(scene)
    validate_analytic_crystal(scene,crystal)
    return crystal


def _packet(cp, checkpoint, fov_xy, coarse_yx, support, field_half=None):
    """Explicit source regridding with independent interpolation evidence."""
    from temsim.physics.specimen_wave_transport import _regrid_mode
    ny, nx = coarse_yx
    x = (np.arange(nx)-nx//2)*fov_xy[0]/nx*1e-10
    y = (np.arange(ny)-ny//2)*fov_xy[1]/ny*1e-10
    fx, fy = np.meshgrid(np.fft.fftfreq(nx, fov_xy[0]/nx), np.fft.fftfreq(ny, fov_xy[1]/ny))
    keep = np.hypot(fx, fy) <= support+2/min(fov_xy)
    gy, gx = np.meshgrid(np.fft.fftfreq(ny, 1/ny).round().astype(int),
                         np.fft.fftfreq(nx, 1/nx).round().astype(int), indexing="ij")
    spectra, weights, receipts, removed = [], [], [], 0.
    for mode in checkpoint.beam.modes:
        from temsim.physics.wave_device import to_host
        host = replace(mode, plane=replace(mode.plane, amplitude=to_host(mode.plane.amplitude)))
        resampled, receipt = _regrid_mode(host, x, y)
        wave = cp.asarray(resampled.plane.full_amplitude(_wavelength(mode.energy_kev*1000)*1e-10))
        if field_half is not None:
            wave = field_half(wave, (fov_xy[0]/nx, fov_xy[1]/ny), cp)
        spectrum = cp.asnumpy(cp.fft.fft2(wave, norm="ortho"))
        loss = float(np.sum(abs(spectrum[~keep])**2))*mode.weight_per_reference_electron
        removed += loss
        spectra.append(spectrum[keep])
        weights.append(mode.weight_per_reference_electron)
        receipts.append(receipt)
    if removed > 1e-10*checkpoint.beam.total_weight:
        raise ValueError("Source regridding or split field exceeds the unchanged 1e-10 Fourier tail gate")
    energy = checkpoint.beam.modes[0].energy_kev*1000
    if any(mode.energy_kev != checkpoint.beam.modes[0].energy_kev for mode in checkpoint.beam.modes):
        raise ValueError("Mixed STEM needs the declared common stationary-energy lift")
    return ScanPacket(np.array(spectra).T, np.array(weights), gx[keep], gy[keep], _wavelength(energy)), {
        "interpolation": receipts, "numerical_source_fourier_tail_fraction": removed,
        "all_density_components_retained": len(weights), "coarse_grid_yx": [ny, nx]}


def _field_model(state, source, fov_xy, max_frequency):
    """Sample/refine the captured static field; retain honest finite-grid scope."""
    from temsim.physics.unified_field_runtime import capture_unified_field_runtime
    from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
    start = source.z_mm*1e-3
    length = float(state.sample.thickness_nm)*1e-9
    runtime = capture_unified_field_runtime(state, start, start+length)
    if type(runtime.snapshot) is DynamicPotentialSnapshot:
        raise ValueError("A stationary mixed STEM scan cannot infer dynamic coherence from particle clocks")
    sample0 = runtime.snapshot.sample(np.array([[0., 0., start]]))
    a0 = sample0["A_t_m"][0]
    if np.linalg.norm(a0) > 1e-20 or np.any(runtime.snapshot.gauge.gradient_t_m) or np.any(runtime.snapshot.gauge.hessian_t):
        raise ValueError("Mixed STEM split currently requires the native zero-axis magnetic gauge; a canonical spectrum is not a mechanical angle chart")
    phi0 = float(sample0["phi_v"][0])
    bz = float(sample0["B_t"][0, 2])
    levels = []
    lam = _wavelength(source.mean_energy_ev)*1e-10
    rest = m_e*c*c/e
    fmax = max_frequency*1e10
    for nz, nxy in ((5, 17), (9, 33)):
        x = np.linspace(-fov_xy[0]/2, fov_xy[0]/2, nxy)*1e-10
        y = np.linspace(-fov_xy[1]/2, fov_xy[1]/2, nxy)*1e-10
        xx, yy = np.meshgrid(x, y)
        xy = np.c_[xx.ravel(), yy.ravel()]
        residual_a, residual_phi, max_a, max_b = 0., 0., 0., 0.
        for z in np.linspace(start, start+length, nz):
            field = runtime.snapshot.sample(np.c_[xy, np.full(len(xy), z)])
            ideal = np.c_[-bz*xy[:, 1]/2, bz*xy[:, 0]/2, np.zeros(len(xy))]
            residual_a = max(residual_a, float(np.max(np.linalg.norm(field["A_t_m"]-ideal, axis=1))))
            residual_phi = max(residual_phi, float(np.max(abs(field["phi_v"]-phi0))))
            max_a = max(max_a, float(np.max(np.linalg.norm(ideal, axis=1))))
            max_b = max(max_b, float(np.max(abs(field["B_t"][:, :2]))))
        alpha = e/h*residual_a
        beta = ((source.mean_energy_ev+residual_phi)*(source.mean_energy_ev+residual_phi+2*rest)
                /(source.mean_energy_ev*(source.mean_energy_ev+2*rest))-1)
        gap = 1-(lam*fmax)**2
        delta = beta+lam**2*(2*fmax*alpha+alpha**2)
        if gap <= delta or gap <= 0:
            raise ValueError("Mixed STEM field remainder exceeds the positive kinetic branch")
        remainder = 2*np.pi*length*(delta/(lam*(math.sqrt(gap)+math.sqrt(gap-delta)))+alpha)
        leading = 2*np.pi*length*lam*(fmax*e/h*max_a+.5*(e/h*max_a)**2)
        # The paraxial magnetic correction multiplies the exact free kinetic
        # operator. Bound its omitted positive-branch correction over the
        # complete retained material disk, not just the incident alpha95.
        nonparaxial = leading*(1/math.sqrt(gap)-1)
        levels.append({"axial_samples": nz, "lateral_samples": nxy,
            "maximum_nonuniform_A_t_m": residual_a, "maximum_phi_change_v": residual_phi,
            "maximum_transverse_B_t": max_b, "nonuniform_phase_bound_rad": remainder,
            "magnetic_nonparaxial_phase_bound_rad": nonparaxial})
    error_phase = 2*max(row["nonuniform_phase_bound_rad"] for row in levels)+max(row["magnetic_nonparaxial_phase_bound_rad"] for row in levels)
    def apply(wave, distance_angstrom, pixel_xy, cp):
        distance = distance_angstrom*1e-10
        angle = e*bz*lam*distance/(2*h)
        wave = rotate_wave(wave, angle, pixel_xy, xp=cp)
        ny, nx = wave.shape[-2:]
        x = (cp.arange(nx)-nx//2)*pixel_xy[0]*1e-10
        y = (cp.arange(ny)-ny//2)*pixel_xy[1]*1e-10
        a_squared = (e*bz/(2*h))**2*(x[None, :]**2+y[:, None]**2)
        return wave*cp.exp(-1j*np.pi*lam*distance*a_squared)
    return apply, {"method": "Symmetric material / axial-Larmor-plus-diamagnetic overlap split",
        "scope": "Finite sampled static-field model; spatial and axial remainder preflight is refined, not a continuum interval certificate",
        "runtime_digest": runtime.digest, "snapshot_digest": runtime.snapshot.numerical_digest,
        "axial_B_t": bz, "entrance_phi_v": phi0, "native_axis_A_t_m": a0.tolist(),
        "sampling_refinement": levels, "finite_model_phase_remainder_bound_rad": error_phase,
        "detector_absolute_error_bound_per_incident_weight": 2*error_phase,
        "total_larmor_rotation_rad": e*bz*lam*length/(2*h),
        "canonical_readout_scope": "Ideal annular Fourier diagnostic in native zero-axis gauge; not a downstream physical recording detector",
        "omitted_field_terms": "Nonuniform magnetic/electric and nonparaxial magnetic remainders bounded above; exact free kinetic retained"}


def _intervals(detectors):
    edges = sorted({0., 1000., *(v for d in detectors for v in (d.inner_mrad, d.outer_mrad))})
    return {f"interval_{i}": pair for i, pair in enumerate(zip(edges[:-1], edges[1:]))}


def _detectors(observed, intervals, detectors):
    return {d.key: observed[[i for i, (lo, hi) in enumerate(intervals.values())
                    if lo >= d.inner_mrad and hi <= d.outer_mrad]].sum(axis=0) for d in detectors}


def _legacy_generic_potential(state, fov_xy, pixels, count, budget):
    from temsim.specimen.scene import SpecimenScene
    from temsim.physics.mixed_stem_material import _legacy_abtem_finite_cif_potential
    scene = SpecimenScene.from_state(state)
    if scene.source_kind == "cif":
        return _legacy_abtem_finite_cif_potential(scene, fov_xy, (pixels, pixels), count,
                                    maximum_working_bytes=budget)
    from temsim.physics.wave_imaging import prepare_specimen_potentials
    from temsim.physics.wave_sampling import PotentialPreparationBudget
    from temsim.specimen.presets import load_specimen_preset
    from temsim.specimen.scene import SpecimenScene
    numerical = copy(state)
    numerical.sample = copy(state.sample)
    numerical.sample.wave_grid_pixels = pixels
    numerical.sample.wave_field_of_view_angstrom = fov_xy[0]
    numerical.sample.wave_slice_thickness_angstrom = float(state.sample.thickness_nm)*10/count
    scene = SpecimenScene.from_state(numerical)
    prepared = prepare_specimen_potentials(numerical, load_specimen_preset(scene.wave_template_key),
        field_of_view_angstrom_override=fov_xy[0],
        preparation_budget=PotentialPreparationBudget(budget,
            propagation_working_bytes=256*pixels*pixels))
    if prepared.metrics.get("atomistic_fallback_reason"):
        raise ValueError("The actual specimen potential could not be prepared: "+str(prepared.metrics["atomistic_fallback_reason"]))
    configs = prepared.potential_configurations_v_angstrom
    if len(configs) != 1 or np.ndim(configs[0]) != 3:
        raise ValueError("Mixed STEM needs one explicit static three-dimensional potential")
    slices = np.asarray(prepared.slice_thicknesses_angstrom, dtype=float)
    if slices.size != len(configs[0]) or not np.isclose(slices.sum(), scene.thickness_nm*10, atol=1e-8, rtol=0):
        raise ValueError("The prepared material slices do not span the physical slab")
    actual_fov = (float((prepared.x_angstrom[1]-prepared.x_angstrom[0])*len(prepared.x_angstrom)),
                  float((prepared.y_angstrom[1]-prepared.y_angstrom[0])*len(prepared.y_angstrom)))
    return np.asarray(configs[0], dtype=np.float32), slices, {"potential_preparation": prepared.metrics,
        "grid_yx": list(np.shape(configs[0])[-2:]), "slices": len(slices),
        "supercell_angstrom_xy": list(actual_fov), "periodic_qualification": False}


def _generic_potential(*_args,**_kwargs):
    raise ValueError("Mixed STEM analytic finite reference requires a proven rectangular quarter-lattice CIF; no old-core fallback is available")


def _material_level_plan(request, fov, wavelength, tiles, thickness_nm):
    """Capture explicit material numerics or the bounded adaptive ladder."""
    divisor = math.lcm(*tiles) if tiles is not None else 2
    if request.material_levels is not None:
        levels = tuple((math.ceil(pixels/divisor)*divisor, count)
                       for pixels, count in request.material_levels)
        if any(a == b for a, b in zip(levels, levels[1:])):
            raise ValueError("Commensurate rounding collapses two requested material refinement levels")
        return levels
    lateral = list(dict.fromkeys([_PAIRS[0][0][0]]+[pair[1][0] for pair in _PAIRS]
                                +[2160, 2240, 2400, 2560]))
    eligible = []
    maximum_detector = max(d.outer_mrad for d in request.detectors)
    for requested in lateral:
        n = math.ceil(requested/divisor)*divisor
        cutoff = request.bandwidth_fraction*min(n/(2*fov[0]), n/(2*fov[1]))
        if math.asin(min(1., wavelength*cutoff))*1000 > maximum_detector and n not in eligible:
            eligible.append(n)
    minimum_slices = max(8, math.ceil(thickness_nm*10/.5))
    return tuple((pixels, minimum_slices*2**index) for index, pixels in enumerate(eligible))


def _scan_memory_plan(pixels, count, batch, nfreq, modes, retained=0,
                      positions_count=0, detector_count=0, tiles=None, shifted=False,basis_atoms=16):
    primitive_pixels = 0 if tiles is None else pixels*pixels//math.prod(tiles)
    outputs = positions_count*(80+16*(detector_count+5))
    device = pixels*pixels*(192*batch+192)+32*nfreq*modes+32*count*primitive_pixels+outputs
    storage={"host_potential_storage_bytes":int(4*count*pixels*pixels)}
    if tiles is not None:
        from temsim.physics.material_slice_storage import periodic_host_storage_plan
        storage=periodic_host_storage_plan(pixels,count,tiles,shifted=shifted,basis_atoms=basis_atoms)
        potential_peak=storage["host_potential_stage_peak_bytes"]
        device=max(device,storage['periodic_builder_device_peak_bytes'])
    else:
        potential_peak=4*count*pixels*pixels
    host = retained+potential_peak+48*nfreq*modes+outputs
    return {"device_working_bytes": int(device), "host_stage_peak_bytes": int(host), **storage,
            "device_potential_policy": "Primitive-period transmissions for Bloch; one streamed full-grid slice for resident"}


def _admit_memory(cp, checkpoint, request, pixels, count, nfreq, modes, positions_count=0, detector_count=0, tiles=None, shifted=False,basis_atoms=16):
    from temsim.physics.host_memory_budget import check_host_allocation
    retained = sum(int(m.plane.amplitude.nbytes) for m in checkpoint.beam.modes)
    # Potential/gratings/kinetic, batched resident fields, FFT scratch, and
    # source factors. Only one candidate's device buffers are live at a time.
    estimate = _scan_memory_plan(pixels, count, request.batch_size, nfreq, modes,
        retained, positions_count, detector_count, tiles, shifted,basis_atoms)
    free, _ = cp.cuda.runtime.memGetInfo()
    device = estimate["device_working_bytes"]
    if device > request.maximum_working_bytes or device > .8*free:
        raise MemoryError(f"Mixed STEM candidate requires {device} device bytes, exceeding its bounded CUDA workspace")
    # Existing source/history allocations are already excluded by the OS's
    # currently available memory. Admit only additional host stage buffers.
    check_host_allocation(estimate["host_stage_peak_bytes"],
                          requested_maximum_bytes=request.maximum_host_working_bytes)
    return estimate


@input_io.using_state_inputs
def estimate_mixed_stem_host_bytes(state, request=MixedStemScanRequest(), *, source=None, positions_nm=None):
    """Cheap host reservation; excludes the caller's retained particle history.

    It uses the same angular/slice ladder and excludes candidates already
    rejected by the nominal device-stage cap. Actual free-device and host
    admissions remain mandatory at execution. Supply the exact source to
    include its measured transverse centre; otherwise the native axis is the
    reservation estimate and a 20% host planning margin is retained.
    """
    from temsim.specimen.scene import SpecimenScene
    request.validate()
    scene = SpecimenScene.from_state(state)
    shift_a = ((request.pixels-1)*request.pitch_nm*5 if positions_nm is None
               else float(np.max(abs(np.asarray(positions_nm))))*10)
    center_a = (0. if source is None else float(np.max(abs(np.average(
        np.c_[source.x_m, source.y_m], weights=source.weight, axis=0))))*1e10)
    width = request.source.width_nm*10*1.02+2*shift_a+2*center_a
    crystal = _rectangular_crystal(scene)
    if crystal is not None:
        tiles = tuple(max(2, math.ceil(width/r/2)*2) for r in crystal[0][:2])
        fov = tuple(crystal[0][:2]*tiles)
    else:
        tiles, fov = None, (width, width)
    wavelength = _wavelength(source.mean_energy_ev if source is not None else float(state.beam_voltage_kv)*1000)
    levels = _material_level_plan(request, fov, wavelength, tiles, scene.thickness_nm)
    modes = request.source.maximum_modes
    checkpoint_bytes = 16*modes*request.source.pixels**2
    nfreq = math.ceil(np.pi*(request.source.frequency_support_inv_angstrom+2/min(fov))**2*np.prod(fov))
    packet_bytes = 64*nfreq*modes
    plans, maximum = [], 2*checkpoint_bytes+packet_bytes
    for index, (n, count) in enumerate(levels):
        memory = _scan_memory_plan(n, count, request.batch_size, nfreq, modes,
                                  checkpoint_bytes, tiles=tiles, shifted=any(scene.centre_xy_nm),
                                  basis_atoms=len(crystal[1]) if crystal is not None else 16)
        nominal_device_peak = memory["device_working_bytes"]
        potential_peak = memory["host_stage_peak_bytes"]+packet_bytes
        active = (nominal_device_peak <= request.maximum_working_bytes
                  and potential_peak <= request.maximum_host_working_bytes)
        if active:
            maximum = max(maximum, potential_peak)
        plans.append({"pixels": n, "slices": count, "nominal_device_stage_bytes": nominal_device_peak,
                      "candidate_active_within_declared_cap": active, "host_peak_bytes": potential_peak})
        if active and index < len(levels)-1:
            ny, nx = n, n
            large = ((ny//tiles[1]*(tiles[1]+2), nx//tiles[0]*(tiles[0]+2)) if tiles is not None
                     else (2*math.ceil(n*1.25/2),)*2)
            from temsim.physics.material_slice_storage import finite_fourier_storage_plan
            if crystal is None:
                # Resource-only estimate for an invalid/unprepared GUI draft.
                # Execution rejects this scope before particle/source work.
                finite_plan={'host_builder_peak_bytes':4*count*np.prod(large),
                             'unsupported_material_preflight_pending':True}
            else:
                large_fov=tuple(crystal[0][:2]*(np.asarray(tiles)+2))
                finite_plan=finite_fourier_storage_plan(scene,crystal,large_fov,large,count)
            finite_host=(finite_plan['host_builder_peak_bytes']+10*8*np.prod(large)
                         +checkpoint_bytes+packet_bytes)
            plans[-1]['finite_reference_memory']=finite_plan
            maximum = max(maximum, int(finite_host))
    return {"estimated_host_peak_bytes": math.ceil(maximum*1.2),
        "retained_particle_history_included": False, "planning_margin_fraction": .2,
        "source_center_from_exact_source": source is not None,
        "source_checkpoint_bytes": checkpoint_bytes, "candidate_plans": plans,
        "source_fov_angstrom_xy": list(fov),
        "policy": "Maximum live host stage, not the sum of candidates; streamed finite-potential builder slabs"}


def _pattern_l1_same_fov(a, b):
    """Compare absolute probabilities at identical reciprocal-cell spacing."""
    shape = np.maximum(np.shape(a), np.shape(b))
    arrays = []
    for value in (a, b):
        pad = shape-np.shape(value)
        if np.any(pad % 2):
            raise ValueError("Raw diffraction frequency grids are not centre aligned")
        arrays.append(np.pad(value, [(int(v//2), int(v//2)) for v in pad]))
    return float(np.sum(abs(arrays[0]-arrays[1])))


def _heldout_actual_positions(positions):
    # Select actual raster extremes, not assumed positions around laboratory
    # zero. This is an observational check, not a proof for every scan pixel.
    directions = np.array([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
    selected = int(np.argmin(np.sum(positions**2, axis=1)))
    indices = list(dict.fromkeys([int(np.argmax(positions @ d)) for d in directions]+[selected]))
    return np.asarray(indices), indices.index(selected), selected


def _potential_error_fraction(plan, source_fraction):
    """Charge certified potential arithmetic/registration/image errors to current."""
    error = plan.get("numerical_potential_error", {})
    per_incident = error.get("detector_absolute_error_bound_per_incident_weight", 0.)
    if (isinstance(per_incident, (bool, np.bool_)) or not isinstance(per_incident, (int, float, np.number))
            or not math.isfinite(float(per_incident)) or float(per_incident) < 0):
        raise ValueError("The finite atomic-potential error bound must be finite and nonnegative")
    return float(per_incident)*float(source_fraction)


def _qualify_domain_and_diffraction(cp, state, scene, source, checkpoint, request,
        positions, accepted, refined, crystal, tiles, sigma, intervals, cancelled, progress):
    """One enlarged finite-specimen reference and unscaled raw-pattern checks."""
    from temsim.physics.mixed_stem_material import finite_cif_potential
    material_options=dict(energy_kev=source.mean_energy_ev/1000,xp=cp,
        maximum_device_working_bytes=request.maximum_working_bytes,
        cancelled=cancelled,progress_callback=progress)
    fov = accepted["fov"]
    ny, nx = accepted["plan"]["grid_yx"]
    count = accepted["plan"]["slices"]
    if crystal is not None:
        enlarged_tiles = tuple(t+2 for t in tiles)
        large_shape = (ny//tiles[1]*enlarged_tiles[1], nx//tiles[0]*enlarged_tiles[0])
        large_fov = tuple(crystal[0][:2]*enlarged_tiles)
    else:
        large_shape = tuple(2*math.ceil(n*1.25/2) for n in (ny, nx))
        large_fov = (fov[0]*large_shape[1]/nx, fov[1]*large_shape[0]/ny)
    _admit_memory(cp, checkpoint, replace(request, batch_size=1), max(large_shape), count,
        len(accepted["raw_packet"].gx), len(accepted["raw_packet"].weights), 5, len(intervals))
    if progress:
        progress(0, 1, "Qualifying enlarged finite-specimen domain and raw exit diffraction")
    indices, selected_local, selected_global = _heldout_actual_positions(positions)
    heldout = positions[indices]
    material, slices, _ = (_periodic_potential(crystal, scene, fov, tiles, accepted["requested_pixels"], count,**material_options)
        if crystal is not None else _generic_potential(state, fov, accepted["requested_pixels"], count, request.maximum_host_working_bytes))
    cutoff = request.bandwidth_fraction*min(nx/(2*fov[0]), ny/(2*fov[1]))
    field_operator, _ = _field_model(state, source, fov, cutoff)
    base_diag = {}
    base_observed, _, base_large_pattern = resident_observe(cp, accepted["raw_packet"], material, slices, fov,
        heldout, intervals, interaction=sigma, bandwidth_fraction=request.bandwidth_fraction,
        batch_size=1, field=field_operator, field_blocks=1, diffraction=True,
        diffraction_grid_shape=large_shape, diffraction_per_position=True, diagnostics=base_diag,
        cancelled=cancelled, progress_callback=progress)
    # The published pattern and the split/grid controls have one actual scan
    # position. No interpolation or display image enters these comparisons.
    _, _, base_pattern = resident_observe(cp, accepted["raw_packet"], material, slices, fov,
        positions[selected_global:selected_global+1], intervals, interaction=sigma,
        bandwidth_fraction=request.bandwidth_fraction, batch_size=1, field=field_operator,
        field_blocks=1, diffraction=True, cancelled=cancelled, progress_callback=progress)
    _, _, base_four_pattern = resident_observe(cp, accepted["raw_packet"], material, slices, fov,
        positions[selected_global:selected_global+1], intervals, interaction=sigma,
        bandwidth_fraction=request.bandwidth_fraction, batch_size=1, field=field_operator,
        field_blocks=4, diffraction=True, cancelled=cancelled, progress_callback=progress)
    del material, slices
    fine_fov = refined["fov"]
    fine_y, fine_x = refined["plan"]["grid_yx"]
    material, slices, _ = (_periodic_potential(crystal, scene, fine_fov, tiles,
        refined["requested_pixels"], refined["plan"]["slices"],**material_options) if crystal is not None else
        _generic_potential(state, fine_fov, refined["requested_pixels"], refined["plan"]["slices"], request.maximum_host_working_bytes))
    if not np.allclose(fine_fov, fov, rtol=0, atol=1e-10):
        raise ValueError("Raw grid/slice diffraction qualification requires a common physical FOV")
    fine_field, _ = _field_model(state, source, fine_fov,
        request.bandwidth_fraction*min(fine_x/(2*fine_fov[0]), fine_y/(2*fine_fov[1])))
    _, _, fine_pattern = resident_observe(cp, refined["raw_packet"], material, slices, fine_fov,
        positions[selected_global:selected_global+1], intervals, interaction=sigma,
        bandwidth_fraction=request.bandwidth_fraction, batch_size=1, field=fine_field,
        field_blocks=4, diffraction=True, cancelled=cancelled, progress_callback=progress)
    del material, slices
    large_material, large_slices, finite_record = finite_cif_potential(scene, large_fov, large_shape, count,
        maximum_working_bytes=request.maximum_host_working_bytes,crystal=crystal,**material_options)
    original_density = max(1e-10/float(np.min(np.linalg.svd(m.plane.basis_m, compute_uv=False))) for m in checkpoint.beam.modes)
    large_coarse = tuple(2*math.ceil(length*original_density/2) for length in (large_fov[1], large_fov[0]))
    large_packet, large_packet_receipt = _packet(cp, checkpoint, large_fov, large_coarse,
        request.source.frequency_support_inv_angstrom)
    large_field, large_field_record = _field_model(state, source, large_fov, cutoff)
    large_diag = {}
    large_observed, _, large_pattern = resident_observe(cp, large_packet, large_material, large_slices, large_fov,
        heldout, intervals, interaction=sigma, bandwidth_fraction=request.bandwidth_fraction,
        batch_size=1, field=large_field, field_blocks=4, diffraction=True,
        diffraction_per_position=True, diagnostics=large_diag, cancelled=cancelled, progress_callback=progress)
    base_d, large_d = [_detectors(value, intervals, request.detectors) for value in (base_observed, large_observed)]
    domain_changes = {d.key: float(np.max(abs(base_d[d.key]-large_d[d.key]))) for d in request.detectors}
    pattern_domain = float(np.sum(abs(base_large_pattern[selected_local]-large_pattern[selected_local])))
    pattern_grid = _pattern_l1_same_fov(base_four_pattern, fine_pattern)
    pattern_split = _pattern_l1_same_fov(base_pattern, base_four_pattern)
    field_remainder = max(accepted["field_absolute"], refined["field_absolute"],
        large_field_record["detector_absolute_error_bound_per_incident_weight"]*large_packet.fraction)
    large_potential_error = _potential_error_fraction(finite_record, large_packet.fraction)
    potential_remainder = (accepted.get("potential_absolute", 0.)
                           +refined.get("potential_absolute", 0.)+large_potential_error)
    remainder = field_remainder+potential_remainder
    numerator = pattern_grid+pattern_split+pattern_domain+remainder
    reference_total = float(np.sum(fine_pattern))
    ratio = numerator/reference_total if reference_total > 0 else math.inf
    record = {"model_scope": "Finite periodic base compared with an enlarged actual finite-CIF specimen; held-out observation, not a general isolated-carrier certificate",
        "base_fov_angstrom_xy": list(fov), "enlarged_fov_angstrom_xy": list(large_fov),
        "base_grid_yx": [ny, nx], "enlarged_grid_yx": list(large_shape),
        "same_physical_pitch": True, "heldout_actual_positions_nm": (heldout*.1).tolist(),
        "selected_diffraction_position_nm": (positions[selected_global]*.1).tolist(),
        "maximum_BF_ADF_domain_change_fraction": domain_changes,
        "finite_specimen_reference": finite_record, "enlarged_source_regridding": large_packet_receipt,
        "base_depth_edges": base_diag, "enlarged_depth_edges": large_diag,
        "raw_diffraction_comparison": {"grid_slice_l1_fraction": pattern_grid,
            "field_split_l1_fraction": pattern_split, "domain_l1_fraction": pattern_domain,
            "field_remainder_absolute_fraction_bound": field_remainder,
            "potential_remainder_absolute_fraction_bound": potential_remainder,
            "combined_l1_fraction_of_emitted": numerator, "reference_total_fraction_of_emitted": reference_total,
            "combined_relative_l1": ratio if math.isfinite(ratio) else None,
            "maximum_relative_l1": .05, "qualified": ratio <= .05,
            "normalization": "Absolute probabilities remain unchanged; only the error metric is divided by one reference total",
            "domain_common_grid": "Actual base complex exit field zero embedded at unchanged real-space pitch into enlarged physical FOV before orthonormal FFT"}}
    # The caller already charged base/refined potential errors. Add only the
    # enlarged reference term there; the raw-pattern budget above needs all.
    return domain_changes, field_remainder+large_potential_error, base_pattern, record


def _execute_scan(cp, state, source, checkpoint, request, positions, crystal, cancelled, progress):
    from temsim.specimen.scene import SpecimenScene
    from temsim.physics.specimen_wave_transport import _covering_domain
    from temsim.physics.wave_imaging import interaction_constant_rad_per_v_angstrom
    scene = SpecimenScene.from_state(state)
    from temsim.physics.mixed_stem_material import validate_analytic_crystal
    validate_analytic_crystal(scene,crystal)
    material_options=dict(energy_kev=source.mean_energy_ev/1000,xp=cp,
        maximum_device_working_bytes=request.maximum_working_bytes,
        cancelled=cancelled,progress_callback=progress)
    center, width_m = _covering_domain(checkpoint.beam)
    # Include the complete source rectangle for every requested position.
    # Extra vacuum is physical domain, not a larger artificial probe.
    width_a = width_m*1e10+2*float(np.max(abs(positions)))+2*float(np.max(abs(center)))*1e10
    if crystal is not None:
        repeat = crystal[0]
        trial_tiles = tuple(max(2, math.ceil(width_a/r/2)*2) for r in repeat[:2])
        trial_fov = repeat[:2]*trial_tiles
        corners = np.array([(x, y) for x in (-trial_fov[0]/20, trial_fov[0]/20)
                            for y in (-trial_fov[1]/20, trial_fov[1]/20)])
        if (not np.all(scene.sample_contains_xy(corners[:, 0], corners[:, 1]))
                or scene.support_grid.material.atomic_number != 0):
            # A periodic crystal does not prove a periodic holder/envelope.
            crystal = None
    if crystal is None:
        fov = (width_a, width_a)
        tiles = None
    else:
        repeat = crystal[0]
        tiles = tuple(max(2, math.ceil(width_a/r/2)*2) for r in repeat[:2])
        fov = tuple(repeat[:2]*tiles)
    original_density = max(1e-10/float(np.min(np.linalg.svd(m.plane.basis_m, compute_uv=False))) for m in checkpoint.beam.modes)
    coarse = tuple(2*math.ceil(length*original_density/2) for length in (fov[1], fov[0]))
    original_packet, source_receipt = _packet(cp, checkpoint, fov, coarse, request.source.frequency_support_inv_angstrom)
    intervals = _intervals(request.detectors)
    sigma = interaction_constant_rad_per_v_angstrom(source.mean_energy_ev/1000)
    wavelength = original_packet.wavelength_angstrom
    qualification = []
    cache = {}
    independent_completed = False
    backend = "periodic-bloch" if crystal is not None else "resident"
    half_distance = scene.thickness_nm*5
    max_detector = max(d.outer_mrad for d in request.detectors)
    selected = None
    levels = _material_level_plan(request, fov, wavelength, tiles, scene.thickness_nm)
    if len(levels) < 2:
        raise ValueError("The bounded material grid plan cannot cover the complete requested annulus on this source/scan domain")
    pairs = tuple(zip(levels[:-1], levels[1:]))
    for base_key, fine_key in pairs:
        _cancelled_check(cancelled)
        for key in (base_key, fine_key):
            if key in cache:
                continue
            pixels, count = key
            if tiles is not None:
                pixels = math.ceil(pixels/math.lcm(*tiles))*math.lcm(*tiles)
            cutoff = request.bandwidth_fraction*min(pixels/(2*fov[0]), pixels/(2*fov[1]))
            maximum_angle = math.asin(min(1., wavelength*cutoff))*1000
            if maximum_angle <= max_detector:
                cache[key] = {"insufficient_bandwidth": True, "maximum_angle_mrad": maximum_angle}
                continue
            try:
                admitted = _admit_memory(cp, checkpoint, request, pixels, count,
                    len(original_packet.gx), len(original_packet.weights), len(positions), len(intervals), tiles,
                    any(scene.centre_xy_nm),len(crystal[1]))
            except MemoryError as error:
                cache[key] = {"budget_rejected": str(error)}
                continue
            if crystal is not None:
                material, slices, plan = _periodic_potential(crystal, scene, fov, tiles, pixels, count,**material_options)
            else:
                material, slices, plan = _generic_potential(state, fov, pixels, count, request.maximum_host_working_bytes)
            local_fov = tuple(plan["supercell_angstrom_xy"])
            local_y, local_x = material.shape[-2:]
            cutoff = request.bandwidth_fraction*min(local_x/(2*local_fov[0]), local_y/(2*local_fov[1]))
            maximum_angle = math.asin(min(1., wavelength*cutoff))*1000
            if maximum_angle <= max_detector:
                raise ValueError("Realised material lattice does not retain the complete angular annulus")
            local_coarse = tuple(2*math.ceil(length*original_density/2) for length in (local_fov[1], local_fov[0]))
            raw_packet, local_source_receipt = (original_packet, source_receipt) if local_fov == fov else _packet(
                cp, checkpoint, local_fov, local_coarse, request.source.frequency_support_inv_angstrom)
            field_operator, field_record = _field_model(state, source, local_fov, cutoff)
            lam_m = wavelength*1e-10
            radius_m = .5*float(np.linalg.norm(local_fov))*1e-10
            scan_shift_m = float(np.max(np.linalg.norm(np.r_[positions, _VALIDATION_NM*10], axis=1)))*1e-10
            diamagnetic_coefficient = (np.pi*lam_m*scene.thickness_nm*.5e-9
                                      *(e*field_record["axial_B_t"]/(2*h))**2)
            output_diamagnetic_phase = abs(diamagnetic_coefficient)*radius_m**2
            input_translation_phase = abs(diamagnetic_coefficient)*(2*radius_m*scan_shift_m+scan_shift_m**2)
            additional_phase = output_diamagnetic_phase+input_translation_phase
            field_record.update(
                final_diamagnetic_omission_phase_bound_rad=output_diamagnetic_phase,
                initial_diamagnetic_translation_commutator_phase_bound_rad=input_translation_phase,
                phase_bound_radius_m=radius_m, phase_bound_maximum_scan_shift_m=scan_shift_m,
                additional_periodic_field_factorization_phase_bound_rad=additional_phase,
                canonical_to_mechanical_angular_uncertainty_bound_mrad=(e*abs(field_record["axial_B_t"])*radius_m*lam_m/(2*h))*1000,
                angular_axis_scope="qx/qy are canonical cycles/angstrom in the captured native gauge; the stated local mechanical-angle uncertainty is not removed by relabeling the axes")
            field_record["detector_absolute_error_bound_per_incident_weight"] += 2*additional_phase
            packet, packet_receipt = _packet(cp, checkpoint, local_fov, local_coarse,
                request.source.frequency_support_inv_angstrom,
                lambda wave, pixel, xp: field_operator(wave, half_distance, pixel, xp))
            angle = field_record["total_larmor_rotation_rad"]/2
            rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            validation_positions = _VALIDATION_NM*10
            rotated_positions = validation_positions @ rotation.T
            plan.update(maximum_admitted_angle_mrad=maximum_angle,
                estimated_working_bytes=max(admitted["host_stage_peak_bytes"], admitted["device_working_bytes"]),
                memory_admission=admitted, kinetic="exact positive-branch square root, complex128",
                potential_storage_dtype=str(material.dtype), transmission_precision="complex128",
                bandwidth_fraction=request.bandwidth_fraction)
            if progress is not None:
                progress(0, 1, f"Qualifying mixed STEM {pixels} pixels / {count} slices")
            if backend == "periodic-bloch":
                harmonic, operator_record = periodic_coefficients(cp, packet, material, slices, local_fov, tiles,
                    intervals, interaction=sigma, bandwidth_fraction=request.bandwidth_fraction,
                    cancelled=cancelled, progress_callback=progress)
                observed = evaluate_periodic(harmonic, rotated_positions, operator_record["period_angstrom_xy"])
                loss = packet.fraction-observed.sum(axis=0)
                if loss.min() < -2e-11:
                    raise ValueError("Bloch output created probability")
                operation = (harmonic, operator_record, rotation)
            else:
                observed, loss, _ = resident_observe(cp, raw_packet, material, slices, local_fov,
                    validation_positions, intervals, interaction=sigma, bandwidth_fraction=request.bandwidth_fraction,
                    batch_size=request.batch_size, field=field_operator, cancelled=cancelled, progress_callback=progress)
                operation = None
            detectors = _detectors(observed, intervals, request.detectors)
            # The first periodic operator is cross-checked independently.
            # Expensive field-block refinement is deferred until a candidate
            # can pass the lateral/slice/remainder budget, without accepting
            # an image until that independent check is complete.
            one = observed[:, :3] if backend == "resident" else None
            independent = None
            if backend == "periodic-bloch" and not independent_completed:
                one, _, _ = resident_observe(cp, raw_packet, material, slices, local_fov,
                    validation_positions[:3], intervals, interaction=sigma, bandwidth_fraction=request.bandwidth_fraction,
                    batch_size=request.batch_size, field=field_operator, field_blocks=1, cancelled=cancelled,
                    progress_callback=progress)
                independent = float(np.max(abs(one-observed[:, :3])))
                independent_completed = True
            if backend == "periodic-bloch" and independent is not None and independent > 5e-7:
                # A failed reduction is not a CUDA-to-CPU fallback. Repeat the
                # candidate using the independently defined CUDA resident path.
                backend = "resident"
                observed, loss, _ = resident_observe(cp, raw_packet, material, slices, local_fov,
                    validation_positions, intervals, interaction=sigma, bandwidth_fraction=request.bandwidth_fraction,
                    batch_size=request.batch_size, field=field_operator, cancelled=cancelled, progress_callback=progress)
                detectors = _detectors(observed, intervals, request.detectors)
                operation = None
                one = observed[:, :3]
            split_error = {d.key: 0. for d in request.detectors}
            field_absolute = field_record["detector_absolute_error_bound_per_incident_weight"]*raw_packet.fraction
            potential_absolute = _potential_error_fraction(plan, raw_packet.fraction)
            # The negligible translation/diamagnetic commutator is bounded
            # rather than restoring norm or pretending it commutes exactly.
            field_record.update(independent_periodic_resident_max_fraction_difference=independent,
                split_block_refinements=[], split_validation_positions_nm=_VALIDATION_NM[:3].tolist(),
                maximum_split_change_fraction=split_error,
                finite_field_absolute_fraction_bound=field_absolute)
            cache[key] = {"observed": observed, "loss": loss, "detectors": detectors,
                "plan": plan, "field": field_record, "operation": operation,
                "split_error": split_error, "field_absolute": field_absolute,
                "potential_absolute": potential_absolute,
                "packet": packet, "packet_receipt": packet_receipt,
                "raw_packet": raw_packet, "fov": local_fov,
                "source_receipt": local_source_receipt, "requested_pixels": pixels,
                "one_block_validation": one, "split_qualified": False}
            del material, slices
            cp.get_default_memory_pool().free_all_blocks()
        base, fine = cache[base_key], cache[fine_key]
        if base.get("budget_rejected") or fine.get("budget_rejected"):
            qualification.append({"base": list(base_key), "refined": list(fine_key),
                "qualified": False, "reason": "Declared candidate memory budget exceeded",
                "base_rejection": base.get("budget_rejected"), "refined_rejection": fine.get("budget_rejected")})
            continue
        if base.get("insufficient_bandwidth") or fine.get("insufficient_bandwidth"):
            qualification.append({"base": list(base_key), "refined": list(fine_key),
                "qualified": False, "reason": "Complete requested annular bandwidth not represented"})
            continue
        contrasts, absolute, ratios = {}, {}, {}
        for detector in request.detectors:
            name = detector.key
            contrasts[name] = float(np.ptp(fine["detectors"][name]))
            sampling = float(np.max(abs(fine["detectors"][name]-base["detectors"][name])))
            absolute[name] = (sampling+base["split_error"][name]+base["field_absolute"]
                              +base.get("potential_absolute", 0.)+fine.get("potential_absolute", 0.))
            ratios[name] = absolute[name]/contrasts[name] if contrasts[name] else math.inf
        passed = all(value <= request.maximum_modulation_error for value in ratios.values())
        if passed and not base["split_qualified"]:
            local_fov = base["fov"]
            pixels, count = base["requested_pixels"], base_key[1]
            material, slices, _ = (_periodic_potential(crystal, scene, local_fov, tiles, pixels, count,**material_options)
                if crystal is not None else _generic_potential(state, local_fov, pixels, count, request.maximum_host_working_bytes))
            local_y, local_x = material.shape[-2:]
            cutoff = request.bandwidth_fraction*min(local_x/(2*local_fov[0]), local_y/(2*local_fov[1]))
            field_operator, _ = _field_model(state, source, local_fov, cutoff)
            one = base["one_block_validation"]
            if one is None:
                one, _, _ = resident_observe(cp, base["raw_packet"], material, slices, local_fov,
                    _VALIDATION_NM[:3]*10, intervals, interaction=sigma, bandwidth_fraction=request.bandwidth_fraction,
                    batch_size=request.batch_size, field=field_operator, field_blocks=1,
                    cancelled=cancelled, progress_callback=progress)
            selected_independent = float(np.max(abs(one-base["observed"][:, :3])))
            base["field"]["selected_periodic_resident_max_fraction_difference"] = selected_independent
            if base["operation"] is not None and selected_independent > 5e-7:
                # Rebuild/qualify the general CUDA resident model. Do not use
                # coefficients whose selected candidate failed its own check.
                fallback = _execute_scan(cp, state, source, checkpoint, request, positions,
                    None, cancelled, progress)
                fallback[-1]["periodic_fallback_reason"] = "Selected periodic candidate failed independent CUDA resident cross-check"
                return fallback
            two, _, _ = resident_observe(cp, base["raw_packet"], material, slices, local_fov,
                _VALIDATION_NM[:3]*10, intervals, interaction=sigma, bandwidth_fraction=request.bandwidth_fraction,
                batch_size=request.batch_size, field=field_operator, field_blocks=2,
                cancelled=cancelled, progress_callback=progress)
            four, _, _ = resident_observe(cp, base["raw_packet"], material, slices, local_fov,
                _VALIDATION_NM[:3]*10, intervals, interaction=sigma, bandwidth_fraction=request.bandwidth_fraction,
                batch_size=request.batch_size, field=field_operator, field_blocks=4,
                cancelled=cancelled, progress_callback=progress)
            values = [_detectors(v, intervals, request.detectors) for v in (one, two, four)]
            for d in request.detectors:
                base["split_error"][d.key] = float(max(np.max(abs(values[0][d.key]-values[2][d.key])),
                    np.max(abs(values[1][d.key]-values[2][d.key]))))
                absolute[d.key] += base["split_error"][d.key]
                ratios[d.key] = absolute[d.key]/contrasts[d.key] if contrasts[d.key] else math.inf
            base["split_qualified"] = True
            base["field"].update(split_block_refinements=[1, 2, 4], maximum_split_change_fraction=base["split_error"])
            passed = all(value <= request.maximum_modulation_error for value in ratios.values())
            del material, slices, one, two, four
        qualification.append({"base": list(base_key), "refined": list(fine_key),
            "peak_to_peak_reference_fraction": contrasts, "combined_absolute_fraction_bound": absolute,
            "combined_error_over_reference_modulation": {key: value if math.isfinite(value) else None for key, value in ratios.items()},
            "qualified": passed, "base_plan": base["plan"], "refined_plan": fine["plan"],
            "field_model": base["field"],
            "base_potential_absolute_fraction_bound": base.get("potential_absolute", 0.),
            "refined_potential_absolute_fraction_bound": fine.get("potential_absolute", 0.),
            "field_split_refinement_executed": base["split_qualified"],
            "refined_candidate_field_split_refinement_executed": fine["split_qualified"]})
        if passed:
            domain_changes, domain_field_bound, raw_pattern, domain_record = _qualify_domain_and_diffraction(
                cp, state, scene, source, checkpoint, request, positions, base, fine,
                crystal, tiles, sigma, intervals, cancelled, progress)
            for d in request.detectors:
                absolute[d.key] += domain_changes[d.key]+max(0., domain_field_bound-base["field_absolute"])
                ratios[d.key] = absolute[d.key]/contrasts[d.key] if contrasts[d.key] else math.inf
            passed = (all(value <= request.maximum_modulation_error for value in ratios.values())
                      and domain_record["raw_diffraction_comparison"]["qualified"])
            qualification[-1].update(qualified=passed,
                combined_absolute_fraction_bound=absolute,
                combined_error_over_reference_modulation={key: value if math.isfinite(value) else None for key, value in ratios.items()},
                domain_and_raw_diffraction=domain_record)
            base["qualified_diffraction"] = raw_pattern
            base["domain_qualification"] = domain_record
        if progress is not None:
            progress(len(qualification), len(pairs), "Mixed STEM combined grid / slice / field qualification")
        if passed:
            selected = base_key
            break
    if selected is None:
        raise MixedStemQualificationError("No mixed STEM candidate meets the unchanged combined 5% reference-modulation budget", qualification)
    accepted = cache[selected]
    if not accepted["split_qualified"]:
        raise ValueError("Mixed STEM cannot publish a candidate with pending field-split qualification")
    fov, original_packet, source_receipt = accepted["fov"], accepted["raw_packet"], accepted["source_receipt"]
    pixels, count = accepted["requested_pixels"], selected[1]
    field_operator, _ = _field_model(state, source, fov,
        request.bandwidth_fraction*min(pixels/(2*fov[0]), pixels/(2*fov[1])))
    material, slices, _ = (_periodic_potential(crystal, scene, fov, tiles, pixels, count,**material_options)
        if crystal is not None else _generic_potential(state, fov, pixels, count, request.maximum_host_working_bytes))
    if accepted["operation"] is not None and backend == "periodic-bloch":
        harmonic, operator, rotation = accepted["operation"]
        observed = evaluate_periodic(harmonic, positions @ rotation.T, operator["period_angstrom_xy"])
        loss = accepted["packet"].fraction-observed.sum(axis=0)
    else:
        observed, loss, _ = resident_observe(cp, original_packet, material, slices, fov,
            positions, intervals, interaction=sigma, bandwidth_fraction=request.bandwidth_fraction,
            batch_size=request.batch_size, field=field_operator, cancelled=cancelled, progress_callback=progress)
    detector_values = _detectors(observed, intervals, request.detectors)
    other = observed.sum(axis=0)-sum(detector_values.values())
    source_loss = source.source_fraction-checkpoint.beam.total_weight
    representation_loss = checkpoint.beam.total_weight-original_packet.fraction
    if np.min(loss) < -2e-11 or representation_loss < -1e-10:
        raise ValueError("Mixed STEM propagation or representation created current")
    fractions = {**detector_values, "other": np.maximum(other, 0.),
        "numerical_band_loss": np.maximum(loss, 0.),
        "source_mode_loss": np.full(len(positions), max(0., source_loss)),
        "source_representation_loss": np.full(len(positions), max(0., representation_loss)),
        "upstream_physical_loss": np.full(len(positions), max(0., 1-source.source_fraction))}
    closure = sum(fractions.values())
    residual = float(np.max(abs(closure-1)))
    if residual > 1e-9:
        raise ValueError("Mixed STEM absolute source/current/loss ledger does not close")
    diffraction = accepted.get("qualified_diffraction") if request.save_diffraction else None
    diffraction_index = int(np.argmin(np.sum(positions**2, axis=1)))
    if request.save_diffraction and diffraction is None:
        _, _, diffraction = resident_observe(cp, original_packet, material, slices, fov,
            positions[diffraction_index:diffraction_index+1], intervals, interaction=sigma,
            bandwidth_fraction=request.bandwidth_fraction, batch_size=1, field=field_operator,
            diffraction=True, cancelled=cancelled, progress_callback=progress)
    ny, nx = material.shape[-2:]
    qx = np.fft.fftshift(np.fft.fftfreq(nx, fov[0]/nx))
    qy = np.fft.fftshift(np.fft.fftfreq(ny, fov[1]/ny))
    return fractions, diffraction, qx, qy, {
        "backend": "CUDA complex128", "material_algorithm": backend,
        "material_level_plan": [list(level) for level in levels],
        "material_levels_explicit": request.material_levels is not None,
        "kernel_scope": "Local exact free angular-spectrum multislice with qualified static near-uniform axial-field symmetric overlap split; not the general complete column kernel",
        "detector_scope": "Ideal native-gauge exit Fourier annuli; no physical detector-position claim",
        "field_model": accepted["field"], "sampling_qualification": qualification,
        "domain_and_raw_diffraction_qualification": accepted.get("domain_qualification"),
        "source_regridding": source_receipt, "material_plan": accepted["plan"],
        "fraction_reference": "original emitted electron; no post-loss renormalization",
        "probability_balance_maximum_absolute_residual": residual,
        "diffraction_scope": "One actual selected scan position; full unscaled probability per reciprocal pixel, not a scan average or 4D-STEM cube",
        "diffraction_position_nm": (positions[diffraction_index]*.1).tolist(),
        "numerical_band_loss_max_fraction": float(np.max(loss)),
        "source_fraction": source.source_fraction, "retained_source_fraction": checkpoint.beam.total_weight,
        "sample_upper_z_mm": source.z_mm, "sample_thickness_nm": scene.thickness_nm,
        "CIF_path": scene.cif_path, "orientation_quaternion_wxyz": list(scene.orientation_quaternion_wxyz)}


def _source_input_handoff(inputs, source, validated_boundary):
    if validated_boundary is None:
        if inputs.digest != source.input_digest:
            raise ValueError("Mixed STEM inputs changed after particle transport; do not relabel the source")
        return {"method": "Current implementation exact particle boundary",
                "particle_input_digest": source.input_digest,
                "wave_input_digest": inputs.digest,
                "wave_implementation": getattr(inputs, "implementation", None)}
    from temsim.physics.particle_wave_source_io import ValidatedParticleBoundary
    if type(validated_boundary) is not ValidatedParticleBoundary:
        raise TypeError("A cross-implementation wave handoff requires a verified ValidatedParticleBoundary")
    original = validated_boundary.inputs
    if validated_boundary.source.digest != source.digest or original.digest != source.input_digest:
        raise ValueError("The validated particle boundary does not own this unchanged source")
    from temsim.physics.wave_input_compatibility import compare_wave_input_graphs
    try:
        graph_compatibility = compare_wave_input_graphs(original.graph, inputs.graph)
    except ValueError as error:
        raise ValueError("Wave inputs differ from the validated particle boundary's captured physical inputs") from error
    if thaw_json(original.external_inputs) != thaw_json(inputs.external_inputs):
        raise ValueError("Wave inputs differ from the validated particle boundary's captured physical inputs")
    return {"method": "Explicit immutable validated particle boundary; new wave implementation, no particle restart",
            "particle_input_digest": source.input_digest,
            "particle_producer_implementation": original.implementation,
            "wave_input_digest": inputs.digest,
            "wave_implementation": inputs.implementation,
            "original_particle_identity_preserved": True,
            "input_graph_compatibility": graph_compatibility,
            "boundary_validation": thaw_json(validated_boundary.record)}


@input_io.using_state_inputs
def simulate_mixed_stem_scan(state, source, request=MixedStemScanRequest(), *,
                            checkpoint=None, positions_nm=None, cancelled=None,
                            progress_callback=None, validated_boundary=None):
    """Calculate a qualified, absolute-current local mixed STEM acquisition."""
    request.validate()
    cancelled = (lambda: False) if cancelled is None else cancelled
    _cancelled_check(cancelled)
    require_coherent_cuda()
    if not isinstance(source, ParticleWaveSource):
        raise TypeError("Mixed STEM requires its executed immutable ParticleWaveSource")
    inputs = capture_instrument_snapshot(state)
    input_handoff = _source_input_handoff(inputs, source, validated_boundary)
    from temsim.simulation_modes import is_unified
    if not is_unified(state):
        raise ValueError("Mixed STEM requires the executed Finite-Field Optics particle source")
    from temsim.specimen.scene import SpecimenScene
    scene = SpecimenScene.from_state(state)
    upper = float(state.sample.z_mm)-float(state.sample.thickness_nm)*.5e-6
    if not math.isclose(source.z_mm, upper, rel_tol=0, abs_tol=4*abs(float(np.spacing(upper)))):
        raise ValueError("Mixed STEM requires the actual specimen upper-surface source")
    if scene.is_vacuum or not scene.structure_available:
        raise ValueError("Mixed STEM acquisition requires an actual inserted material structure")
    crystal=preflight_mixed_stem_material(state)
    if np.ptp(source.kinetic_energy_ev) > 2.:
        raise ValueError("The stationary mean-wavelength source approximation exceeds its unchanged 2 eV span guard")
    if not bool(getattr(state.sample, "wave_multislice_enabled", True)):
        raise ValueError("A finite-thickness mixed STEM scan requires actual multislice; a phase object is not substituted")
    if bool(getattr(state.sample, "real_high_angle_tail_enabled", False)):
        raise ValueError("The selected separate high-angle material channel cannot be omitted by mixed STEM")
    if bool(getattr(state.sample, "wave_frozen_phonon_enabled", False)):
        raise ValueError("This local mixed STEM entry currently requires the static elastic potential; frozen-phonon histories cannot be omitted")
    from temsim.specimen.inelastic import real_inelastic_distribution
    material_distribution = real_inelastic_distribution(state)
    if any(math.isfinite(value) for value in (material_distribution.total_inelastic_mean_free_path_nm,
                                             material_distribution.absorption_mean_free_path_nm)):
        raise ValueError("Selected finite material inelastic/absorption channels require their distributed loss histories; this elastic scan cannot discard them")
    if checkpoint is None:
        checkpoint = build_particle_wave_checkpoint(source, request.source, cancelled=cancelled, progress_callback=progress_callback)
    if (not isinstance(checkpoint, WaveCheckpoint) or checkpoint.record.get("particle_source_digest") != source.digest
            or checkpoint.record.get("input_digest") != source.input_digest or checkpoint.plane_z_mm != source.z_mm):
        raise ValueError("Mixed STEM checkpoint does not belong to the exact particle source")
    if thaw_json(checkpoint.record.get("numerics", {})) != asdict(request.source):
        raise ValueError("Mixed STEM checkpoint coherence/numerics differ from the requested local source")
    if (checkpoint.reference_current_a != source.reference_current_a
            or checkpoint.record.get("original_positive_count") != source.original_positive_count
            or checkpoint.record.get("handoff_particle_count") != int(np.count_nonzero(source.weight > 0))
            or not math.isclose(float(checkpoint.record.get("source_fraction", -1)), source.source_fraction, rel_tol=1e-14, abs_tol=0)
            or any(not math.isclose(mode.energy_kev*1000, source.mean_energy_ev, rel_tol=1e-14, abs_tol=0)
                   for mode in checkpoint.beam.modes)):
        raise ValueError("Mixed STEM checkpoint current, original population or stationary energy differs from the particle source")
    positions, shape, scan_x, scan_y = _positions(request, positions_nm)
    started = perf_counter()
    cp = cupy_module()
    result = _execute_scan(cp, state, source, checkpoint, request, positions, crystal,
                           cancelled, progress_callback)
    _cancelled_check(cancelled)
    if capture_instrument_snapshot(state).digest != inputs.digest:
        raise ValueError("Mixed STEM inputs changed during acquisition; no image published")
    fractions, diffraction, qx, qy, record = result
    fractions = {key: np.asarray(value).reshape(shape) for key, value in fractions.items()}
    mean_lambda = _wavelength(source.mean_energy_ev)
    largest_angle = max(d.outer_mrad for d in request.detectors)*1e-3
    frequency = math.sin(largest_angle)/mean_lambda
    distance = scene.thickness_nm*10
    def axial_phase(wavelength):
        u = (wavelength*frequency)**2
        return -2*np.pi*distance/wavelength*u/(1+math.sqrt(1-u))
    energy_limits = (float(source.kinetic_energy_ev.min()), float(source.kinetic_energy_ev.max()))
    wavelength_limits = [_wavelength(energy) for energy in energy_limits]
    energy_approximation = {
        "method": "One current-weighted stationary mean wavelength; actual per-particle energies retained in source ancestry",
        "energy_mode_integration_executed": False,
        "source_configured_fwhm_ev": float(getattr(state.electron_gun.emitter, "energy_spread_fwhm_ev", 0.)),
        "actual_mean_energy_ev": source.mean_energy_ev,
        "actual_energy_limits_ev": list(energy_limits),
        "actual_energy_span_ev": float(np.ptp(source.kinetic_energy_ev)),
        "mean_wavelength_angstrom": mean_lambda,
        "maximum_free_phase_change_rad_at_outer_detector": max(abs(axial_phase(w)-axial_phase(mean_lambda)) for w in wavelength_limits),
        "maximum_angular_axis_change_mrad_at_outer_detector": max(abs(math.asin(w*frequency)-largest_angle)*1000 for w in wavelength_limits),
        "scope": "Free propagation and angular-coordinate sensitivity only; not a full material energy-mode error certificate"}
    record.update(schema="mixed-particle-stem-scan-v1", input_digest=inputs.digest,
        particle_input_handoff=input_handoff,
        source_digest=source.digest, checkpoint_digest=checkpoint.digest,
        request={**asdict(request), "detectors": [asdict(d) for d in request.detectors]},
        reference_current_a=source.reference_current_a, source_reference=checkpoint.beam.reference_plane,
        original_positive_count=source.original_positive_count,
        surviving_positive_count=int(np.count_nonzero(source.weight > 0)),
        effective_surviving_count=source.source_fraction**2/float(source.weight@source.weight),
        position_shape=list(shape), positions_nm=(positions*.1).tolist(),
        elapsed_s=perf_counter()-started, tip_coherent_executed=False,
        material_inelastic_scope={"model": material_distribution.model,
            "material_key": material_distribution.material_key,
            "zero_loss_removal_rate_per_nm": 0., "warnings": list(material_distribution.warnings),
            "scope": "Configured elastic branch; unavailable IMFP data are not a prediction of absent real inelastic scattering"},
        wave_energy_approximation=energy_approximation,
        source_numerical_record=thaw_json(checkpoint.record))
    return MixedStemScanResult(np.asarray(scan_x), np.asarray(scan_y), fractions,
        {key: value*source.reference_current_a for key, value in fractions.items()},
        diffraction, qx, qy, checkpoint, freeze_json(record))


__all__ = ("MixedStemScanRequest", "MixedStemScanResult", "MixedStemQualificationError",
           "estimate_mixed_stem_host_bytes", "simulate_mixed_stem_scan")

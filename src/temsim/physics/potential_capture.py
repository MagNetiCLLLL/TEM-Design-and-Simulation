"""Worker-side capture of explicitly selected native column potentials.

This opt-in P1 entry point preserves the solved A_phi, not an inverse of a
sampled B map. It does not replace legacy transport or qualify a coherent
operator. Selection is explicit: gun alignment coils, blankers, filters,
specimens and unknown maps are not implicitly represented by zero fields.
"""
import json
from types import SimpleNamespace

from temsim import input_io
from temsim.component_keys import CONDENSER_LENS_KEYS
from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.axisymmetric_potential import AxisymmetricMagneticPotential
from temsim.physics.lens_field_provider import (
    FrozenAnalyticField, GeometryAwareAnalyticFieldProvider,
    MappedLensFieldProvider, resolve_runtime_lens_field_provider,
)
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.potential_snapshot import (
    ElectromagneticPotentialSnapshot, PotentialContribution,
)


def _keys(values, label):
    if (not isinstance(values, (tuple, list)) or any(not isinstance(key, str) or not key
            for key in values) or len(set(values)) != len(values)):
        raise ValueError(f"{label} must be a sequence of unique nonempty keys")
    return tuple(values)


@input_io.using_state_inputs
def capture_round_lens_potential(state, key):
    """Capture one selected lens or its inseparable joint nonlinear circuit.

    Resolving a configured FEM recipe may solve it on a cache miss. Call from
    a numerical worker, never a plot refresh. The legacy B interpolant is not
    changed; this representation evaluates curl of the original P1 FEM basis.
    """
    lenses = {lens.key: lens for lens in getattr(state, "lenses", ())}
    if key not in lenses or not bool(getattr(lenses[key], "enabled", True)):
        raise ValueError(f"Unknown or disabled round lens: {key}")
    native = state.condenser_system[key] if key in CONDENSER_LENS_KEYS else lenses[key]
    provider = resolve_runtime_lens_field_provider(state, key, native)
    if type(provider) is GeometryAwareAnalyticFieldProvider:
        from temsim.test_electron_compiled_laws import magnetic_provider_is_supported, lens_family, _LENSES
        admitted_native = lens_family(native) is not None
        if not admitted_native:
            from temsim.optics.image_corrector import ImageCorrectorTl31LensComponent, ImageCorrectorTl32LensComponent
            from temsim.optics.round_lens import RoundLensComponent
            admitted_native = (type(native) in (ImageCorrectorTl31LensComponent, ImageCorrectorTl32LensComponent)
                and all(getattr(getattr(native, name), "__func__", None) is original
                        for name, original in _LENSES[RoundLensComponent][1]))
        if not magnetic_provider_is_supported(provider, allow_rigid_pose=True) or not admitted_native:
            raise ValueError(f"{key}: native analytical field law is not admitted for potential capture")
        potential = AnalyticMagneticPotential(FrozenAnalyticField.from_provider(provider), {
            "model_status": provider.model_status,
            "geometry_fingerprint": provider.binding.geometry_fingerprint,
            "fallback_reason": provider.fallback_reason,
        })
        return PotentialContribution(key, (key,), potential, "magnetic")
    if type(provider) is not MappedLensFieldProvider:
        raise ValueError(f"{key}: unsupported potential provider")
    descriptor = getattr(state, "lens_field_map_descriptors", {}).get(key, {})
    solver = descriptor.get("solver")
    from temsim.physics.axisymmetric_magnetostatics import SOLVER_VERSION, solve_geometry_field_details
    if solver == "axisymmetric_nonlinear_fem":
        from temsim.physics.nonlinear_circuits import resolve_nonlinear_provider
        geometry, settings, owners = resolve_nonlinear_provider(
            state, key, native, provider.binding, prepare_only=True)
        binding = SimpleNamespace(canonical_geometry_json=json.dumps(geometry, sort_keys=True, allow_nan=False))
        source_id = "joint_nonlinear:"+"+".join(owners)
    elif solver == "axisymmetric_linear_fem":
        binding, settings, owners, source_id = provider.binding, descriptor, (key,), key
    else:
        raise ValueError(f"{key}: B-only maps have no retained potential; provide original potential coefficients")
    field_map, solution = solve_geometry_field_details(binding, settings)
    if field_map.provenance.source_sha256 != provider.field_map.provenance.source_sha256:
        raise ValueError(f"{key}: retained potential does not match the resolved field solve")
    scale = provider.excitation_scale()
    potential = AxisymmetricMagneticPotential.from_solution(solution,
        registration=provider.field_map.registration, source_id=source_id, source_scale=scale,
        provenance={"solver": SOLVER_VERSION, "solve_source_sha256": field_map.provenance.source_sha256,
                    "geometry": json.loads(binding.canonical_geometry_json), "settings": settings,
                    "relative_residual": solution.relative_residual, "iterations": solution.iterations,
                    "peak_material_t": solution.peak_material_t,
                    "legacy_B_basis": "separately interpolated nodal field; intentionally unchanged",
                    "potential_B_basis": "curl of retained triangular P1 A_phi",
                    "excitation_scaling": provider.excitation_scaling})
    return PotentialContribution(source_id, owners, potential, "magnetic")


@input_io.using_state_inputs
def capture_selected_column_potentials(state, *, round_lens_keys=(), auxiliary_keys=(),
                                       include_electric=False, include_gun_alignment=False, time_s=None):
    """Capture only requested sources; this is not an all-column field solver.

    Auxiliary keys select native stigmators, corrector multipoles or physical
    deflectors. A shared logical drive must be selected through its physical
    host so its coil is captured once. Dynamic drives are frozen at time_s;
    induced electric fields and time evolution are outside this static record.
    Every selected source must admit a query before a sum can be returned.
    """
    round_lens_keys = _keys(round_lens_keys, "Round lenses")
    auxiliary_keys = _keys(auxiliary_keys, "Auxiliary components")
    if type(include_electric) is not bool or type(include_gun_alignment) is not bool:
        raise ValueError("Electric and gun alignment selection must be boolean")
    rows, joint_ids = [], set()
    for key in round_lens_keys:
        row = capture_round_lens_potential(state, key)
        if row.source_id not in joint_ids:
            rows.append(row)
            joint_ids.add(row.source_id)
    if auxiliary_keys:
        from temsim.physics.posed_column_fields import capture_posed_column_fields
        fields = capture_posed_column_fields(state, include_aligned=True, time_s=time_s,
                                            component_keys=auxiliary_keys)
        selected = tuple(field for field in fields if field.component_key in auxiliary_keys)
        missing = set(auxiliary_keys)-{field.component_key for field in selected}
        if missing:
            raise ValueError("Unknown, disabled or unsupported auxiliary potential (select shared physical host): "
                             +", ".join(sorted(missing)))
        for field in selected:
            potential = AnalyticMagneticPotential(field, {"component_key": field.component_key,
                "captured_time_s": field.captured_time_s, "dynamic_drive_frozen": field.dynamic,
                "induced_electric_field_modelled": False})
            rows.append(PotentialContribution(field.lens_key, (field.lens_key,), potential, "magnetic"))
    if include_electric:
        from temsim.physics.instrument_electric import capture_instrument_electric_field
        from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
        potential = ElectricPotentialSnapshot.from_provider(capture_instrument_electric_field(state))
        rows.append(PotentialContribution("instrument_electric", ("instrument_electric",), potential, "electric"))
    if include_gun_alignment:
        from temsim.physics.gun_magnetic_potential import GunMagneticPotential
        gun = getattr(state, "electron_gun", None)
        if gun is None or any(getattr(gun, name, None) is None for name in ("deflector", "stigmator")):
            raise ValueError("This gun has no native alignment pair and stigmator to capture")
        for name in ("deflector", "stigmator"):
            potential = GunMagneticPotential.from_provider(getattr(gun, name))
            rows.append(PotentialContribution(potential.source_id, (potential.source_id,), potential, "magnetic"))
    return ElectromagneticPotentialSnapshot(capture_physical_scene(state), tuple(rows))

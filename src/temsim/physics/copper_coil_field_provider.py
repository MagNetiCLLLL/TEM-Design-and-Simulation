"""Resolved installed air-core windings, with explicit owner excitation.

This source model ignores magnetic pole material; it never fits a Gaussian or
infers ampere-turns from an old optical calibration. Every physical winding
interval retains its own finite support as matter, and its field has full tails.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
import json
import math

import numpy as np

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.physics.annular_coil_potential import AnnularCoilPotential


@dataclass(frozen=True, slots=True)
class CopperCoilAssemblyPotential:
    source_id: str
    coils: tuple
    provenance: Mapping = field(default_factory=dict)
    digest: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id or self.source_id.strip() != self.source_id:
            raise ValueError("Copper coil owner must be a nonempty identifier")
        coils = tuple(self.coils)
        if not coils or any(type(coil) is not AnnularCoilPotential for coil in coils):
            raise ValueError("Copper coil assembly requires explicit immutable annular windings")
        if len({coil.source_id for coil in coils}) != len(coils):
            raise ValueError("Copper coil winding identities must be unique")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("Copper coil provenance must be a mapping")
        object.__setattr__(self, "coils", coils)
        object.__setattr__(self, "provenance", freeze_json(self.provenance))
        object.__setattr__(self, "digest", json_digest(self.to_metadata()))

    @property
    def lens_key(self):
        return self.source_id

    @property
    def fingerprint(self):
        return self.digest

    def to_metadata(self):
        return dict(schema="installed-copper-coil-potential-v1", source_id=self.source_id,
            coils=[coil.to_dict() for coil in self.coils], provenance=thaw_json(self.provenance),
            axial_domain="all_z", empirical_aberrations_applied=False, pole_piece_magnetism=False,
            derivative_order=2, wave_admission=False)

    def to_dict(self):
        return dict(metadata=self.to_metadata(), digest=self.digest)

    @classmethod
    def from_dict(cls, record):
        if not isinstance(record, Mapping) or set(record) != {"metadata", "digest"}:
            raise ValueError("Invalid copper coil assembly snapshot")
        m = record["metadata"]
        if not isinstance(m, Mapping) or m.get("schema") != "installed-copper-coil-potential-v1":
            raise ValueError("Unknown copper coil assembly schema")
        try:
            result = cls(m["source_id"], tuple(AnnularCoilPotential.from_dict(row) for row in m["coils"]), m["provenance"])
        except (TypeError, KeyError) as error:
            raise ValueError("Invalid copper coil assembly coefficients") from error
        if result.digest != record["digest"] or json_digest(m) != result.digest:
            raise ValueError("Copper coil assembly snapshot identity mismatch")
        return result

    def _sum(self, method, positions_m):
        result = None
        for coil in self.coils:
            values = getattr(coil, method)(positions_m)
            if result is None:
                result = values
            else:
                for name, value in values.items():
                    result[name] += value
        return result

    def sample(self, positions_m):
        return self._sum("sample", positions_m)

    def sample_jet(self, positions_m):
        return self._sum("sample_jet", positions_m)

    def field_at_global_positions_t(self, positions_m):
        values = [coil.field_at_global_positions_t(positions_m) for coil in self.coils]
        return sum(values[1:], values[0])

    def require_global_box(self, positions_m):
        for coil in self.coils:
            coil.require_global_box(positions_m)


@dataclass(frozen=True, slots=True)
class CopperCoilFieldProvider:
    lens_key: str
    potential: CopperCoilAssemblyPotential
    binding: object
    model_status: str = "air_core_uniform_copper_windings_no_pole_magnetism"

    def magnetic_field_t(self, z_mm):
        z = np.asarray(z_mm, float)
        positions = np.zeros((*z.shape, 3))
        positions[..., 2] = z*.001
        return self.potential.field_at_global_positions_t(positions)[..., 2]

    def field_at_global_positions_t(self, positions_m):
        return self.potential.field_at_global_positions_t(positions_m)

    def field_support_mm(self, *_args):
        # Scheduling envelope only; no field sampler truncates at these faces.
        from temsim.physics.lens_field_provider import _posed_axial_support
        bounds = [_posed_axial_support(((coil.center_z_m-coil.length_m/2)*1e3,
            (coil.center_z_m+coil.length_m/2)*1e3), coil.registration, coil.outer_radius_m)
            for coil in self.potential.coils]
        return min(row[0] for row in bounds), max(row[1] for row in bounds)


def resolve_copper_coil_provider(state, key, native, binding, registration):
    """Use the same normalized material intervals as the magnetic geometry."""
    from temsim.physics.lens_field_provider import _runtime_excitation
    parts = json.loads(binding.canonical_geometry_json)["lens_assembly"]["parts"]
    owner = next((part for part in parts if part["key"] == key), None)
    if owner is None:
        raise ValueError(f"{key}: Copper Coil requires resolved installed owner geometry")
    reference = owner["data"].get("air_core_reference_ampere_turns")
    if isinstance(reference, bool) or not isinstance(reference, (int, float)) or not math.isfinite(reference) or reference <= 0:
        raise ValueError(f"{key}: Copper Coil requires explicit positive air_core_reference_ampere_turns")
    enabled, percent, polarity = _runtime_excitation(native)
    if not math.isfinite(percent) or not 0 <= percent <= 100 or polarity not in (-1, 1):
        raise ValueError(f"{key}: Copper Coil excitation must be 0..100 percent with polarity +/-1")
    total = reference*percent*.01*polarity if enabled else 0.
    windings = [part for part in parts if part["data"].get("mechanical_profile") == "magnetic_excitation_coil"
                and part["data"].get("field_source_key") == key]
    if not windings:
        raise ValueError(f"{key}: Copper Coil has no explicitly owned winding; Gaussian fallback is disabled")
    from temsim.lens_pose import has_nonrigid_lens_cad
    participating = {key, *(part["key"] for part in windings)}
    unsupported = [part.key for part in getattr(getattr(state, "_resolved_assembly", None), "parts", ())
                   if part.key in participating and has_nonrigid_lens_cad(part.data)]
    if unsupported:
        raise ValueError("Annular copper windings cannot represent non-rigid CAD shape edits on "
                         + ", ".join(unsupported))
    fractions, coils = [], []
    for part in windings:
        data = part["data"]
        fraction = data.get("coil_ampere_turn_fraction")
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or fraction <= 0:
            raise ValueError(f"{part['key']}: Copper Coil requires an explicit positive ampere-turn fraction")
        fractions.append(fraction)
        intervals = data.get("material_intervals_mm", ((part["start_z_mm"], part["end_z_mm"]),))
        previous, length = -math.inf, 0.
        for start, stop in intervals:
            if not math.isfinite(start) or not math.isfinite(stop) or stop <= start or start < previous:
                raise ValueError(f"{part['key']}: Winding intervals must be ordered, disjoint and finite")
            previous, length = stop, length+stop-start
        if not intervals or length <= 0:
            raise ValueError(f"{part['key']}: Empty current-bearing winding")
        try:
            diameters = [data[name] for name in ("mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm")]
        except (KeyError, TypeError) as error:
            raise ValueError(f"{part['key']}: Winding inner and outer diameters are required") from error
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value <= 0 for value in diameters):
            raise ValueError(f"{part['key']}: Winding diameters must be finite positive lengths")
        inner, outer = (value*.0005 for value in diameters)
        for index, (start, stop) in enumerate(intervals):
            coils.append(AnnularCoilPotential(f"{key}:{part['key']}:{index}", inner, outer,
                (stop-start)*.001, total*fraction*(stop-start)/length,
                (start+stop)*.0005, registration))
    if not math.isclose(math.fsum(fractions), 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError(f"{key}: Winding ampere-turn fractions must sum to one")
    resolved_owner = next((part for part in getattr(getattr(state, "_resolved_assembly", None), "parts", ())
                           if part.key == key), None)
    owner_metadata = {} if resolved_owner is None else resolved_owner.data
    provenance = dict(geometry_fingerprint=binding.geometry_fingerprint,
        reference_ampere_turns_at_100_percent=reference, enabled=enabled,
        excitation_percent=percent, polarity=polarity,
        reference_status=owner_metadata.get("air_core_reference_status", "unspecified"),
        reference_source=owner_metadata.get("air_core_reference_source", ""),
        reference_source_urls=owner_metadata.get("air_core_reference_source_urls", ()),
        source_registration_policy="Project optical registration: owner translations and assembly azimuth place the magnetic source; optical tilts rotate its source axis. Material CAD follows its separate mechanical registration; an optical tilt does not imply a mechanically tilted copper winding.",
        current_distribution="Coil fractions per owner; within each coil, uniform radial-axial current density and NI proportional to interval length",
        winding_material="Copper, relative permeability one; pole pieces ignored magnetically")
    potential = CopperCoilAssemblyPotential(key, tuple(coils), provenance)
    cache = getattr(state, "_runtime_copper_coil_provider_cache", {})
    prior = cache.get(key)
    if prior is not None and prior.potential.digest == potential.digest:
        return prior
    provider = CopperCoilFieldProvider(key, potential, binding)
    cache[key] = provider
    state._runtime_copper_coil_provider_cache = cache
    return provider

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
            raise ValueError("Ideal-current coil owner must be a nonempty identifier")
        coils = tuple(self.coils)
        if not coils or any(type(coil) is not AnnularCoilPotential for coil in coils):
            raise ValueError("Ideal-current coil assembly requires explicit immutable annular windings")
        if len({coil.source_id for coil in coils}) != len(coils):
            raise ValueError("Ideal-current coil winding identities must be unique")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("Ideal-current coil provenance must be a mapping")
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
            raise ValueError("Invalid ideal-current coil assembly snapshot")
        m = record["metadata"]
        if not isinstance(m, Mapping) or m.get("schema") != "installed-copper-coil-potential-v1":
            raise ValueError("Unknown ideal-current coil assembly schema")
        try:
            result = cls(m["source_id"], tuple(AnnularCoilPotential.from_dict(row) for row in m["coils"]), m["provenance"])
        except (TypeError, KeyError) as error:
            raise ValueError("Invalid ideal-current coil assembly coefficients") from error
        if result.digest != record["digest"] or json_digest(m) != result.digest:
            raise ValueError("Ideal-current coil assembly snapshot identity mismatch")
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

    def vacuum_mask(self, positions_m):
        valid = np.ones(np.shape(positions_m)[:-1], dtype=bool)
        for coil in self.coils:
            local = coil.registration.positions_global_to_local_m(positions_m)
            valid &= ~coil.contains_winding_local(local)
        return valid


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
        return -math.inf, math.inf


def resolve_copper_coil_provider(state, key, native, binding, registration):
    """Use the same normalized material intervals as the magnetic geometry."""
    from temsim.physics.lens_field_provider import _runtime_excitation
    parts = json.loads(binding.canonical_geometry_json)["lens_assembly"]["parts"]
    owner = next((part for part in parts if part["key"] == key), None)
    if owner is None:
        raise ValueError(f"{key}: Ideal prescribed-current coils require resolved installed owner geometry")
    from temsim.coil_reference_metadata import OWNER_CURRENT_FIELD, round_coil_operating_point
    from temsim.coil_winding import (
        resolve_wire_cross_section_mm2, resolve_wire_diameter_mm,
        validate_interval_turn_counts,
    )
    owner_metadata = state._resolved_assembly.part(key).data
    explicit_current = OWNER_CURRENT_FIELD in owner["data"]
    reference = owner_metadata.get("air_core_reference_ampere_turns")
    if not explicit_current and (isinstance(reference, bool) or not isinstance(reference, (int, float)) or not math.isfinite(reference) or reference <= 0):
        raise ValueError(f"{key}: Ideal prescribed-current coils require explicit positive air_core_reference_ampere_turns or defined turns and current")
    enabled, percent, polarity = _runtime_excitation(native)
    if not math.isfinite(percent) or not 0 <= percent <= 100 or polarity not in (-1, 1):
        raise ValueError(f"{key}: Ideal-current coil excitation must be 0..100 percent with polarity +/-1")
    total = None if explicit_current else reference*percent*.01*polarity if enabled else 0.
    windings = [part for part in parts if part["data"].get("mechanical_profile") == "magnetic_excitation_coil"
                and part["data"].get("field_source_key") == key]
    if not windings:
        raise ValueError(f"{key}: Ideal-current coils have no explicitly owned winding; Gaussian fallback is disabled")
    from temsim.ideal_coil_segments import FIELD, normalize_recipe, bind_segments
    raw_segments = getattr(state, "lens_field_map_descriptors", {}).get(key, {}).get(FIELD)
    segment_recipe = normalize_recipe(raw_segments) if raw_segments is not None else None
    if segment_recipe is not None:
        unknown = set(segment_recipe["windings"])-{part["key"] for part in windings}
        if unknown:
            raise ValueError(f"{key}: Internal currents refer to absent winding(s): "+", ".join(sorted(unknown)))
    operating = (round_coil_operating_point(owner, windings, percent=percent,
                 polarity=polarity, enabled=enabled) if explicit_current else None)
    if operating is not None:
        reference = operating["reference_ampere_turns"]
        total = operating["ampere_turns"]
    from temsim.lens_pose import has_nonrigid_lens_cad
    participating = {key, *(part["key"] for part in windings)}
    unsupported = [part.key for part in getattr(getattr(state, "_resolved_assembly", None), "parts", ())
                   if part.key in participating and has_nonrigid_lens_cad(part.data)]
    if unsupported:
        raise ValueError("Ideal annular windings cannot represent non-rigid CAD shape edits on "
                         + ", ".join(unsupported))
    fractions, coils, segment_records = [], [], []
    for part in windings:
        data = part["data"]
        fraction = (int(data["winding_turn_count"])/operating["total_turn_count"]
                    if operating is not None else data.get("coil_ampere_turn_fraction"))
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or fraction <= 0:
            raise ValueError(f"{part['key']}: Ideal-current coils require an explicit positive ampere-turn fraction")
        fractions.append(fraction)
        intervals = data.get("material_intervals_mm", ((part["start_z_mm"], part["end_z_mm"]),))
        previous, length = -math.inf, 0.
        for start, stop in intervals:
            if not math.isfinite(start) or not math.isfinite(stop) or stop <= start or start < previous:
                raise ValueError(f"{part['key']}: Winding intervals must be ordered, disjoint and finite")
            previous, length = stop, length+stop-start
        if not intervals or length <= 0:
            raise ValueError(f"{part['key']}: Empty current-bearing winding")
        interval_turns = validate_interval_turn_counts(data, interval_count=len(intervals))
        try:
            diameters = [data[name] for name in ("mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm")]
        except (KeyError, TypeError) as error:
            raise ValueError(f"{part['key']}: Winding inner and outer diameters are required") from error
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value <= 0 for value in diameters):
            raise ValueError(f"{part['key']}: Winding diameters must be finite positive lengths")
        inner, outer = (value*.0005 for value in diameters)
        cells = None
        if segment_recipe is not None and part["key"] in segment_recipe["windings"]:
            counts = interval_turns
            if counts is None:
                if len(intervals) != 1 or "winding_turn_count" not in data:
                    raise ValueError(f"{part['key']}: Internal currents need fixed turns in every original interval")
                counts = (int(data["winding_turn_count"]),)
            cells = bind_segments(segment_recipe, part["key"], counts)
        for index, (start, stop) in enumerate(intervals):
            if interval_turns is not None and operating is not None:
                segment_ni = interval_turns[index]*operating["current_a"]
            elif interval_turns is not None and any(
                    count*length != sum(interval_turns)*(b-a)
                    for count, (a,b) in zip(interval_turns, intervals)):
                segment_ni = total*fraction*(interval_turns[index]/sum(interval_turns))
            else:
                # Retain legacy arithmetic when fixed counts already match the
                # original uniform density; recording turn evidence alone must
                # not perturb a historical field's floating-point coefficients.
                segment_ni = (int(data["winding_turn_count"])*operating["current_a"]
                    if operating is not None else total*fraction)*(stop-start)/length
            selected = tuple(cell for cell in cells or () if cell.interval_index == index)
            if not selected:
                coils.append(AnnularCoilPotential(f"{key}:{part['key']}:{index}", inner, outer,
                    (stop-start)*.001, segment_ni,
                    (start+stop)*.0005, registration))
                continue
            physical_count = sum(cell.turn_count for cell in selected)
            master_current = (operating["current_a"] if operating is not None
                              else segment_ni/physical_count)
            uniform = all(cell.current_ratio == selected[0].current_ratio for cell in selected)
            if uniform:
                # Retain the original quadrature and arithmetic for an
                # electrically uniform interval, including exact ratio=1.
                coils.append(AnnularCoilPotential(f"{key}:{part['key']}:{index}", inner, outer,
                    (stop-start)*.001, segment_ni*selected[0].current_ratio,
                    (start+stop)*.0005, registration))
            for segment_index, cell in enumerate(selected):
                r0 = inner+(outer-inner)*cell.radial_fraction[0]
                r1 = inner+(outer-inner)*cell.radial_fraction[1]
                z0 = start+(stop-start)*cell.axial_fraction[0]
                z1 = start+(stop-start)*cell.axial_fraction[1]
                current = master_current*cell.current_ratio
                ni = cell.turn_count*current
                if not all(math.isfinite(value) for value in (current, ni)):
                    raise ValueError(f"{part['key']}: Internal prescribed current exceeds the finite numerical range")
                if not uniform:
                    coils.append(AnnularCoilPotential(
                        f"{key}:{part['key']}:{index}:current_segment:{segment_index}",
                        r0, r1, (z1-z0)*.001, ni, (z0+z1)*.0005, registration))
                segment_records.append(dict(winding_key=part["key"],
                    **cell.to_dict(), inner_radius_m=r0, outer_radius_m=r1,
                    start_z_mm=z0, end_z_mm=z1, current_a=current, ampere_turns=ni,
                    uniform_interval_collapsed=uniform))
    if not math.isclose(math.fsum(fractions), 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError(f"{key}: Winding ampere-turn fractions must sum to one")
    provenance = dict(geometry_fingerprint=binding.geometry_fingerprint,
        reference_ampere_turns_at_100_percent=reference, enabled=enabled,
        excitation_percent=percent, polarity=polarity,
        reference_status=owner_metadata.get("air_core_reference_status", "unspecified"),
        reference_source=owner_metadata.get("air_core_reference_source", ""),
        reference_source_urls=owner_metadata.get("air_core_reference_source_urls", ()),
        source_registration_policy="Project optical registration: owner translations and assembly azimuth place the magnetic source; optical tilts rotate its source axis. Material CAD follows its separate mechanical registration; an optical tilt does not imply a mechanically tilted copper winding.",
        current_distribution="Coil fractions per owner; fixed interval turns when declared, otherwise NI proportional to interval length; uniform current density inside each interval",
        winding_material="Copper, relative permeability one; pole pieces ignored magnetically")
    if segment_recipe is not None:
        provenance[FIELD] = segment_recipe
        provenance["internal_current_distribution"] = dict(
            model="Closed azimuthal current groups inside each unchanged original allocation",
            cells=segment_records,
            segmented_physical_turn_count=sum(row["turn_count"] for row in segment_records),
            signed_segment_ampere_turns=math.fsum(row["ampere_turns"] for row in segment_records),
            absolute_segment_ampere_turns=math.fsum(abs(row["ampere_turns"]) for row in segment_records),
            geometry_changed=False, conductor_area_changed=False,
            resistance_model=False, constant_power_constraint=False,
            current_rule="I_segment = current_ratio * original winding master current at the active control",
            scope="Ideal prescribed-current coils in vacuum only; numerical cells are not additional physical turns")
        provenance["current_distribution"] = "Signed current groups inside original winding intervals; each group's physical turns and full closed loops retained"
    if operating is not None:
        # CAD wire/insulation detail never shrinks the homogenized source.
        # Each winding's NI remains uniformly distributed over its original
        # radial-axial envelope, including any insulation allocation.
        provenance["legacy_reference_record"] = {
            "ampere_turns_at_100_percent": owner_metadata.get("air_core_reference_ampere_turns"),
            "status": provenance["reference_status"],
            "source": provenance["reference_source"],
            "source_urls": provenance["reference_source_urls"],
            "active": False,
        }
        provenance["reference_status"] = "user_defined_winding_current_not_oem_not_particle_qualified"
        provenance["reference_source"] = "Explicit saved turn counts and prescribed electrical current; not an inferred or measured winding calibration"
        provenance["reference_source_urls"] = ()
        provenance["current_distribution"] = "Each owned winding has N times prescribed series current; fixed interval turns when declared, otherwise split in proportion to occupied axial length; uniform within each original interval"
        winding_records = []
        for part in windings:
            data = part["data"]
            inner, outer = (float(data[name]) for name in
                            ("mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"))
            intervals = data.get("material_intervals_mm", ((part["start_z_mm"], part["end_z_mm"]),))
            area = (outer-inner)/2*math.fsum(float(b)-float(a) for a,b in intervals)
            turns = int(data["winding_turn_count"])
            wire = resolve_wire_diameter_mm(data)
            wire_area = resolve_wire_cross_section_mm2(data)
            fixed_intervals = validate_interval_turn_counts(data, interval_count=len(intervals))
            equivalent_current = turns*operating["current_a"]
            winding_records.append(dict(key=part["key"], turn_count=turns,
                interval_turn_counts=fixed_intervals,
                wire_diameter_mm=wire, wire_cross_section_mm2=wire_area,
                mean_diameter_mm=(inner+outer)/2,
                equivalent_current_a=equivalent_current,
                homogenized_cross_section_mm2=area,
                homogenized_current_density_a_per_mm2=equivalent_current/area,
                interval_homogenized_current_density_a_per_mm2=(
                    [count*operating["current_a"]/((outer-inner)/2*(b-a))
                     for count,(a,b) in zip(fixed_intervals,intervals)]
                    if fixed_intervals is not None else [equivalent_current/area]*len(intervals)),
                conductor_cross_section_mm2=(turns*wire_area if wire_area is not None else None)))
        provenance["winding_current_definition"] = {
            **operating, "homogenization": "Original winding envelope; uniform inside each interval with fixed counts when declared; CAD insulation ignored electromagnetically",
            "windings": winding_records,
        }
        if segment_recipe is not None:
            provenance["winding_current_definition"]["internal_current_distribution"] = provenance["internal_current_distribution"]
            provenance["winding_current_definition"]["homogenization"] = "Original envelope tiled by closed signed-current cells; unsegmented intervals retain their original uniform master current"
            provenance["current_distribution"] = "Signed current groups inside unchanged original winding intervals; each cell retains its assigned physical turns and closed azimuthal loops"
    potential = CopperCoilAssemblyPotential(key, tuple(coils), provenance)
    # The unified runtime caches the complete frozen graph. Storing these
    # derived providers on State would pollute snapshots and worker copies.
    return CopperCoilFieldProvider(key, potential, binding)

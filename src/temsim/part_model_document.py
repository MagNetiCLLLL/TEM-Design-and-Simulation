"""Transactional drafts of existing TOML dimensions and material regions."""

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real
from pathlib import Path
from temsim import input_io
import tomllib

from temsim import module_manifest
from temsim.manifest_editor import resized_part_axial_coordinates
from temsim.component_representation import SHARED_DEFLECTOR_HOSTS, shared_deflector_field_owner
from temsim.optics.field_hardware_binding import bound_field_owner, resolve_field_hardware_parts
from temsim.component_operations import (
    PartChangeSet, added_component_document, placed_component_document,
    copied_component_document, validate_component_graph, CUSTOM_COMPONENT_ROLES,
    resized_copied_axial_fields,
)


_WINDING_INPUT_UNSET = object()
PROBE_WINDING_SOURCE_NOTE = (
    "Explicit user-authored engineering radial-winding CAD. Per-coil turns and conductor metal area "
    "are not measured or OEM data. The six radial coils use rounded rectangular closed turns and "
    "automatic remaining insulation inside the existing winding envelope. Native multipole strength "
    "is retained; no conductor-current or NI-to-field calibration is inferred.")
TRANSVERSE_WINDING_SOURCE_NOTE = (
    "Explicit engineering transverse-winding input based on public saddle-coil or radial stigmator "
    "construction families; not measured or OEM winding data. Conductor metal area and insulation "
    "repack turns within the unchanged winding envelope. The ideal vacuum source excludes magnetic "
    "cores and pole pieces. Manual changes do not automatically recalibrate current or refocus a beam.")
MULTIPOLE_WINDING_SOURCE_NOTE = (
    "Explicit engineering multipole winding reconstruction, not measured or OEM hardware data. "
    "Integer per-coil turns, conductor metal area and radial insulation fit the existing fixed "
    "winding allocation. Auxiliary winding-only CAD does not add a cartridge or alter the existing "
    "transport boundaries. Ideal prescribed currents exclude magnetic cores and pole pieces. "
    "Manual edits never refit current, change native optical commands or refocus a beam.")


def _invalidate_multipole_geometry_calibrations(previous, candidate):
    """Keep electrical gains fixed and label a changed multipole fit stale."""
    from temsim.multipole_current import MATRIX_FIELD, NOTE_FIELD, STATUS_FIELD
    from temsim.multipole_winding import (
        multipole_winding_envelopes, multipole_winding_owner_key, turn_field,
    )
    old_parts={part['key']:part for part in previous.get('parts',())}
    new_parts={part['key']:part for part in candidate.get('parts',())}
    reference_fields=('effective_length_mm','max_strength_m2','maximum_strength_m2',
        'maximum_strength_m3','optical_reference_local_z_mm',
        'channel_x_angle_deg','channel_y_angle_deg')

    def reference(part):
        model=part.get('model',{})
        params=model.get('params',{}) if isinstance(model,Mapping) else {}
        if not isinstance(params,Mapping):params={}
        return (tuple((field,part.get(field)) for field in reference_fields),
                tuple((field,params.get(field)) for field in reference_fields))

    for part in candidate.get('parts',()):
        old=old_parts.get(part['key'],{})
        if MATRIX_FIELD not in old or part.get(MATRIX_FIELD)!=old[MATRIX_FIELD]:continue
        owner=multipole_winding_owner_key(part)
        old_owner=multipole_winding_owner_key(old)
        changed=(part.get(turn_field(part))!=old.get(turn_field(old))
            or multipole_winding_envelopes(part,new_parts)!=multipole_winding_envelopes(old,old_parts)
            or reference(part)!=reference(old) or owner!=old_owner
            or reference(new_parts.get(owner,{}))!=reference(old_parts.get(old_owner,{})))
        if not changed:continue
        part[STATUS_FIELD]='stale_geometry_or_turn_edit'
        warning=('The original multipole field-strength match is stale after a manual turn, '
            'envelope or reference-field edit. Numerical current matrix and biases are retained; '
            'no automatic current fit or optical retuning was performed.')
        prior=str(part.get(NOTE_FIELD,''))
        if warning not in prior:part[NOTE_FIELD]=prior+('\n' if prior else '')+warning
        metadata=deepcopy(part.get('parameter_metadata',{}))
        entry=deepcopy(metadata.get(MATRIX_FIELD,{}))
        entry.update(calibration_status='stale_geometry_or_turn_edit',source_note=part[NOTE_FIELD])
        entry.setdefault('source_kind','user_defined')
        metadata[MATRIX_FIELD]=entry
        part['parameter_metadata']=metadata


def _invalidate_transverse_geometry_calibrations(previous, candidate):
    """Retain fixed currents while making an edited geometry's old fit explicit."""
    from temsim.transverse_current import MATRIX_FIELD, NOTE_FIELD, STATUS_FIELD
    from temsim.transverse_winding import (
        TURN_FIELD, transverse_winding_envelopes, transverse_winding_owner_key,
    )
    old_parts={part['key']:part for part in previous.get('parts',())}
    new_parts={part['key']:part for part in candidate.get('parts',())}
    reference_fields=('effective_thickness_mm','effective_length_mm','max_strength_m2',
        'maximum_strength_m2','channel_x_angle_deg','channel_y_angle_deg',
        'optical_reference_local_z_mm','interaction_centers_local_z_mm',
        'active_length_mm','coil_length_mm','soft_edge_mm')

    def reference(part):
        # Native commands are normal operating inputs to the fixed matrix.
        # Only its original field-strength/plane definition invalidates a fit.
        model=part.get('model',{})
        params=model.get('params',{}) if isinstance(model,Mapping) else {}
        if not isinstance(params,Mapping):
            params={}
        return (tuple((field,part.get(field)) for field in reference_fields),
                tuple((field,params.get(field)) for field in reference_fields))

    for part in candidate.get('parts',()):
        old=old_parts.get(part['key'],{})
        if MATRIX_FIELD not in old or part.get(MATRIX_FIELD)!=old[MATRIX_FIELD]:
            continue
        owner_key=transverse_winding_owner_key(part)
        old_owner_key=transverse_winding_owner_key(old)
        changed=(part.get(TURN_FIELD)!=old.get(TURN_FIELD)
            or transverse_winding_envelopes(part)!=transverse_winding_envelopes(old)
            or reference(part)!=reference(old)
            or owner_key!=old_owner_key
            or reference(new_parts.get(owner_key,{}))!=reference(old_parts.get(old_owner_key,{})))
        if not changed:
            continue
        part[STATUS_FIELD]='stale_geometry_or_turn_edit'
        warning=('The original field-strength match is stale after a manual turn, envelope or '
                 'reference-field geometry edit. Numerical current matrix and biases are retained; '
                 'no automatic current fit or optical retuning was performed.')
        prior=str(part.get(NOTE_FIELD,''))
        if warning not in prior:
            part[NOTE_FIELD]=prior+('\n' if prior else '')+warning
        metadata=deepcopy(part.get('parameter_metadata',{}))
        entry=deepcopy(metadata.get(MATRIX_FIELD,{}))
        entry['calibration_status']='stale_geometry_or_turn_edit'
        entry['source_note']=part[NOTE_FIELD]
        entry.setdefault('source_kind','user_defined')
        metadata[MATRIX_FIELD]=entry
        part['parameter_metadata']=metadata


def _manual_winding_source(part, field):
    """Replace only the edited input's provenance inside an atomic candidate."""
    metadata = deepcopy(part.get("parameter_metadata", {}))
    if field not in part:
        metadata.pop(field, None)
    else:
        note = ("Explicit manual winding input. Previous estimates or field-strength "
                "calibrations are not asserted by this edit; no automatic turn estimation, "
                "current calibration or geometry resizing was performed.")
        if field in {"winding_wire_diameter_mm", "winding_wire_cross_section_mm2"}:
            note = ("Explicit manual bare-conductor size. This input changes detailed "
                    "CAD packing only; the ideal current source retains its original envelope.")
        metadata[field] = dict(source_kind="user_defined", source_note=note)
    part["parameter_metadata"] = metadata


def _invalidate_current_calibration(owner, field):
    """Retain current provenance while making a changed-turn fit obsolete."""
    metadata = deepcopy(owner.get("parameter_metadata", {}))
    entry = deepcopy(metadata.get(field, {}))
    if entry.get("calibration_status") != "invalidated_by_manual_turn_edit":
        previous = entry.get("source_note", "")
        warning = ("Field-strength calibration invalidated by a subsequent manual turn-count "
                   "or interval-allocation edit. The prescribed current is retained unchanged; "
                   "no field match is asserted and no current recalibration was performed.")
        entry["source_note"] = previous + ("\n" if previous else "") + warning
    entry.setdefault("source_kind", "user_defined")
    entry["calibration_status"] = "invalidated_by_manual_turn_edit"
    metadata[field] = entry
    owner["parameter_metadata"] = metadata


def _validate_internal_details(parts, geometry=None):
    """Validate the final material draft before publishing it or its history."""
    from temsim.cad_auxiliary import validate_gun_electrode_geometry
    from temsim.pole_geometry import validate_pole_face_land_geometry
    from temsim.cad_recording_planes import validate_detector_footprint
    from temsim.cad_detectors import validate_detector_material_geometry
    from temsim.cad_gun_stigmator import gun_stigmator_dimensions
    from temsim.cad_condenser_cartridge import validate_cartridge_geometry
    from temsim.cad_stigmator_support import stigmator_support_dimensions
    from temsim.cad_aperture_support import validate_aperture_support
    from temsim.cad_c3_lens_hardware import validate_c3_lens_hardware
    from temsim.cad_adapter_lens_hardware import validate_adapter_lens_hardware
    from temsim.cad_probe_transfer_lens_hardware import validate_probe_transfer_lens_hardware
    from temsim.cad_tl22_lens_hardware import validate_tl22_lens_hardware
    from temsim.cad_objective_upper_hardware import validate_objective_upper_hardware
    from temsim.cad_probe_hexapole_support import validate_hexapole_support
    from temsim.cad_probe_winding import validate_probe_winding
    from temsim.cad_multipoles import pole_root_sleeve_solids, validate_auxiliary_return_sleeve
    from temsim.magnetic_geometry import validate_corrector_return_extent
    from temsim.transverse_winding import validate_transverse_winding
    from temsim.transverse_current import validate_transverse_current_inputs
    from temsim.multipole_winding import validate_multipole_winding
    from temsim.multipole_current import validate_multipole_current_inputs
    from temsim.gun_rear_geometry import rear_part_dimensions
    by_key = {part["key"]: part for part in parts}
    from temsim.cad_round_winding import resolve_round_winding_geometry
    for part in parts:
        resolve_round_winding_geometry(part, by_key)
        rear_part_dimensions(part)
        validate_gun_electrode_geometry(part)
        validate_pole_face_land_geometry(part)
        validate_detector_footprint(part)
        validate_detector_material_geometry(part, by_key)
        gun_stigmator_dimensions(part)
        validate_cartridge_geometry(part, by_key)
        stigmator_support_dimensions(part)
        validate_aperture_support(part, by_key, geometry)
        validate_c3_lens_hardware(part, by_key)
        validate_adapter_lens_hardware(part, by_key)
        validate_tl22_lens_hardware(part, by_key)
        validate_objective_upper_hardware(part, by_key)
        validate_probe_transfer_lens_hardware(part, by_key)
        validate_hexapole_support(part)
        pole_root_sleeve_solids(part)
        validate_auxiliary_return_sleeve(part, by_key)
        validate_corrector_return_extent(part)
        validate_probe_winding(part)
        validate_transverse_winding(part,by_key)
        validate_transverse_current_inputs(part,by_key)
        validate_multipole_winding(part,by_key)
        validate_multipole_current_inputs(part,by_key)


class PartModelDocument:
    """A file-backed draft shared by every selected part in the 3D editor."""

    def __init__(self, path):
        self._new_document = False
        self.path = Path(path).resolve()
        self._source_bytes = input_io.read_bytes(self.path)
        self._archive_identity = input_io.active_archive().identity if input_io.active_archive() is not None else None
        from temsim.shared_tip import materialized_text, dependencies
        self._dependency_bytes = dependencies(self.path)
        self._editable_text = materialized_text(self._source_bytes.decode("utf-8-sig"), self.path)
        self._baseline = resolve_field_hardware_parts(tomllib.loads(self._editable_text))
        if not isinstance(self._baseline.get("parts"), list) or not self._baseline["parts"]:
            raise ValueError("Open an instrument module TOML containing [[parts]] definitions")
        keys = [part.get("key") for part in self._baseline["parts"]]
        if any(not isinstance(key, str) or not key for key in keys) or len(keys) != len(set(keys)):
            raise ValueError("Every part must have a unique non-empty key")
        from temsim.probe_hexapole_geometry import validate_probe_hexapole_geometry
        from temsim.corrector_model_3d import validate_deflector_coil_part
        from temsim.ordinary_deflector_geometry import validate_ordinary_deflector_part
        for part in self._baseline["parts"]:
            validate_probe_hexapole_geometry(part)
            validate_deflector_coil_part(part)
            validate_ordinary_deflector_part(part)
        _validate_internal_details(self._baseline["parts"], self._baseline.get("geometry"))
        from temsim.deflector_support_geometry import validate_support_installations
        validate_support_installations(self._baseline["parts"])
        by_key = {part["key"]: part for part in self._baseline["parts"]}
        for key in keys:
            seen = set()
            ancestor = key
            while ancestor in by_key:
                if ancestor in seen:
                    raise ValueError("Component ownership must not contain a cycle")
                seen.add(ancestor)
                ancestor = by_key[ancestor].get("parent_key")
        self.document = deepcopy(self._baseline)
        from temsim.subassemblies import ownership
        self._subassembly_owners = ownership(self._baseline, self.path)
        self._history = [deepcopy(self.document)]
        self._history_index = 0

    @property
    def dirty(self):
        return getattr(self, "_new_document", False) or self.document != self._baseline

    @classmethod
    def from_step_asset(cls, asset, path, *, key=None, name=None):
        """Create an unsaved independent mechanical module, without a temp file."""
        import tomli_w
        from temsim.step_assets import base_from_asset
        base_from_asset(asset)
        result = cls.__new__(cls)
        result.path = Path(path).resolve()
        if result.path.exists():
            raise ValueError("Choose a new TOML path; existing definitions are not replaced by import")
        result._new_document = True
        result._source_bytes = b""
        result._archive_identity = None
        result._dependency_bytes = {}
        result._subassembly_owners = {}
        result._baseline = {"format_version": 1, "coordinate_system": "module_local_z_mm",
                            "module": {"key": "mechanical:step", "type": "mechanical", "name": "Imported STEP"},
                            "geometry": {"length_mm": 1.0}, "parts": []}
        result._editable_text = tomli_w.dumps({k: v for k, v in result._baseline.items() if k != "parts"})
        result.document = deepcopy(result._baseline)
        result._history = [deepcopy(result.document)]
        result._history_index = 0
        result.import_step_asset(asset, key=key, name=name)
        return result

    def import_step_asset(self, asset, *, mode="add", target_key=None,
                          registration="preserve", key=None, name=None):
        """Atomically bind imported material; optical definitions remain intact.

        New components preserve STEP module coordinates. Replacement can centre
        the imported bounding box on the existing part reference; this is an
        explicit registration, never a recovered optical datum from STEP.
        """
        import numpy as np
        from temsim.step_assets import base_from_asset, asset_bounds
        from temsim.part_model_features import default_model_3d, validate_model_3d
        from temsim.component_operations import make_component
        if mode not in {"add", "replace"} or registration not in {"preserve", "center"}:
            raise ValueError("Choose add/replace and preserve/center registration")
        base = base_from_asset(asset)
        center, size = asset_bounds(asset)
        if mode == "add":
            if key is None:
                used = {p["key"] for p in self.document["parts"]}
                index = 1
                while f"step_import_{index}" in used:
                    index += 1
                key = f"step_import_{index}"
            target = make_component(key=key, name=name or asset.get("source", {}).get("filename", key), shape="box",
                                    center_z_mm=float(center[2]), length_mm=float(size[2]),
                                    width_mm=float(size[0]), height_mm=float(size[1]))
            model = default_model_3d(target)
            model["base"] = base
            model["transform"]["offset_mm"] = [float(center[0]), float(center[1]), 0.]
            target["model_3d"] = model
            validate_model_3d(target)
            return self.add_component(target)
        if target_key is None:
            raise ValueError("Select the material component to replace")
        from temsim.component_representation import non_material_role
        target = deepcopy(self.part(target_key))
        if non_material_role(target) or any(p.get("parent_key") == target_key and not non_material_role(p)
                                             for p in self.document["parts"]):
            raise ValueError("Replace a material leaf, not an optical channel or an assembly with child parts")
        model = default_model_3d(target)
        model["base"] = base
        if registration == "preserve":
            from temsim.lens_pose import mechanical_part_transform_mm, supports_physical_lens_pose
            from temsim.energy_filter_model_3d import supports_energy_filter_part
            rotation, offset = mechanical_part_transform_mm(target, {p["key"]: p for p in self.document["parts"]})
            if (supports_physical_lens_pose(target) or supports_energy_filter_part(target)
                    or not np.allclose(rotation, np.eye(3), atol=1e-14, rtol=0)
                    or not np.allclose(offset, np.zeros(3), atol=1e-14, rtol=0)):
                raise ValueError("This component has an optical or branch placement. Choose explicit centre registration to retain that datum")
            model["transform"]["offset_mm"] = [float(center[0]), float(center[1]),
                                                     float(center[2]) - float(target["local_center_z_mm"])]
        # Preserve legacy optical pose before installing new shape-only CAD.
        from temsim.lens_pose import supports_physical_lens_pose, migrate_legacy_lens_pose
        if supports_physical_lens_pose(target):
            migrate_legacy_lens_pose(target)
        target["model_3d"] = model
        validate_model_3d(target)
        self._synchronize_custom_base_length(target)
        self._commit_part(target_key, target)
        return target_key

    def reset_model_geometry(self, key):
        """Restore the imported source size and clear cuts, retaining placement."""
        from temsim.step_assets import read_asset, base_from_asset
        model = deepcopy(self.part(key).get("model_3d", {}))
        if model.get("base", {}).get("kind") != "step":
            self.set_model_3d(key, None)
            return
        model["base"] = base_from_asset(read_asset(model["base"]["asset_json"]))
        model["features"] = []
        model["transform"]["scale_xy"] = [1., 1.]
        self.set_model_3d(key, model)

    @property
    def can_undo(self):
        return self._history_index > 0

    @property
    def can_redo(self):
        return self._history_index + 1 < len(self._history)

    def part(self, key):
        return next(part for part in self.document["parts"] if part["key"] == key)

    def _remember(self):
        if self.document == self._history[self._history_index]:
            return
        self._history = self._history[:self._history_index + 1]
        self._history.append(deepcopy(self.document))
        if len(self._history) > 100:
            self._history.pop(0)
        self._history_index = len(self._history) - 1

    def _commit_part(self, key, candidate):
        from temsim.ordinary_deflector_geometry import validate_ordinary_deflector_part
        from temsim.optics.electron_gun.tip_assembly import complete_tip_updates, validate_tip_part
        from temsim.corrector_model_3d import validate_deflector_coil_part
        from temsim.probe_hexapole_geometry import validate_probe_hexapole_geometry
        previous = self.part(key)
        updates = {("parts", key, field): value for field, value in candidate.items()
                   if previous.get(field) != value}
        for path, value in complete_tip_updates(self.document, updates).items():
            candidate[path[2]] = value
        validate_tip_part(candidate)
        validate_ordinary_deflector_part(candidate)
        validate_deflector_coil_part(candidate)
        validate_probe_hexapole_geometry(candidate)
        for index, part in enumerate(self.document["parts"]):
            if part["key"] == key:
                updated = deepcopy(self.document)
                updated["parts"][index] = candidate
                from temsim.mechanical_dimension_bindings import FIELDS, synchronize_dimension_edit
                for field in FIELDS:
                    if field in part and field in candidate and candidate[field] != part[field]:
                        synchronize_dimension_edit(updated, key, field, part[field])
                from temsim.deflector_support_geometry import validate_support_installations
                validate_support_installations(updated["parts"])
                from temsim.subassemblies import reflow
                candidate_document = reflow(self.document, updated, self.path, owners=self._subassembly_owners)
                candidate_document = resolve_field_hardware_parts(candidate_document)
                _invalidate_transverse_geometry_calibrations(self.document,candidate_document)
                _invalidate_multipole_geometry_calibrations(self.document,candidate_document)
                _validate_internal_details(candidate_document["parts"], candidate_document.get("geometry"))
                from temsim.gun_rear_geometry import validate_rear_gun_document
                validate_rear_gun_document(candidate_document)
                self.document = candidate_document
                self._remember()
                return
        raise ValueError(f"Unknown part: {key}")

    def _commit_component_operation(self, candidate):
        # Structural edits must be valid before replacing any part of the
        # current draft; validation failure preserves both draft and history.
        from temsim.subassemblies import reflow
        candidate = reflow(self.document, candidate, self.path, owners=self._subassembly_owners)
        _invalidate_transverse_geometry_calibrations(self.document,candidate)
        _invalidate_multipole_geometry_calibrations(self.document,candidate)
        validate_component_graph(candidate)
        module_manifest.validate_document(candidate)
        self.document = resolve_field_hardware_parts(candidate)
        self._remember()

    def configure_winding(self, coil_key, *, winding_turn_count,
                          winding_wire_diameter_mm=_WINDING_INPUT_UNSET, winding_interval_turn_counts=_WINDING_INPUT_UNSET,
                          winding_wire_cross_section_mm2=_WINDING_INPUT_UNSET,
                          ideal_coil_current_a_at_100_percent):
        """Stage fixed physical turns and the owner's current atomically.

        The existing ID/OD/axial envelope and optical controls are retained.
        Omitted conductor size retains its current definition. Explicit None
        clears both area and legacy diameter. Omitted counts preserve a split;
        A single interval follows its identical total turn count; explicit None
        removes the split. Invalid inputs leave history unchanged.
        """
        values = dict(winding_turn_count=winding_turn_count)
        if winding_wire_cross_section_mm2 is not _WINDING_INPUT_UNSET:
            values["winding_wire_cross_section_mm2"] = winding_wire_cross_section_mm2
        if winding_wire_diameter_mm is not _WINDING_INPUT_UNSET:
            values["winding_wire_diameter_mm"] = winding_wire_diameter_mm
        if winding_interval_turn_counts is not _WINDING_INPUT_UNSET:
            values["winding_interval_turn_counts"] = winding_interval_turn_counts
        return self.configure_windings(
            self.part(coil_key).get("field_source_key"),
            {coil_key: values},
            ideal_coil_current_a_at_100_percent=ideal_coil_current_a_at_100_percent)

    def configure_probe_winding(self, key, *, turn_count_per_coil, wire_cross_section_mm2):
        """Stage radial HP1/HP2 CAD inputs without defining a field current."""
        from temsim.multipole_current import MATRIX_FIELD as MULTIPOLE_MATRIX
        from temsim.multipole_winding import FIELDS as MULTIPOLE_FIELDS, INSULATION_FIELD
        existing=self.part(key)
        if MULTIPOLE_MATRIX in existing or MULTIPOLE_FIELDS.intersection(existing):
            return self.configure_multipole_winding(key,turn_count_per_coil=turn_count_per_coil,
                wire_cross_section_mm2=wire_cross_section_mm2,
                insulation_thickness_mm=existing.get(INSULATION_FIELD) if turn_count_per_coil is not None else None)
        from temsim.cad_probe_winding import AREA_FIELD, FIELDS, NOTE_FIELD, STATUS_FIELD, TURN_FIELD
        from temsim.probe_hexapole_geometry import is_probe_hexapole
        candidate = deepcopy(self.document)
        part = next((p for p in candidate['parts'] if p['key'] == key), None)
        if (part is None or not is_probe_hexapole(part)
                or part.get('model_3d', {}).get('base', {}).get('kind', 'existing') != 'existing'):
            raise ValueError('Select a native material HP1 or HP2 hexapole')
        if turn_count_per_coil is None and wire_cross_section_mm2 is not None:
            raise ValueError('Conductor metal area requires a known per-coil turn count')
        metadata = deepcopy(part.get('parameter_metadata', {}))
        if turn_count_per_coil is None:
            for field in FIELDS:
                part.pop(field, None)
                metadata.pop(field, None)
        else:
            from temsim.cad_probe_winding import validate_probe_winding
            proposed = {**part, TURN_FIELD: turn_count_per_coil,
                STATUS_FIELD: 'engineering_reconstruction_not_oem', NOTE_FIELD: PROBE_WINDING_SOURCE_NOTE}
            if wire_cross_section_mm2 is None:
                proposed.pop(AREA_FIELD, None)
            else:
                proposed[AREA_FIELD] = wire_cross_section_mm2
            validate_probe_winding(proposed)
            if part.get(TURN_FIELD) == turn_count_per_coil and part.get(AREA_FIELD) == wire_cross_section_mm2:
                return  # Preserve an unchanged saved recipe and its original provenance.
            previous = deepcopy(part)
            part.update(proposed)
            if wire_cross_section_mm2 is None:
                part.pop(AREA_FIELD, None)
                metadata.pop(AREA_FIELD, None)
            for field in (TURN_FIELD, AREA_FIELD):
                if field in part and previous.get(field) != part[field]:
                    metadata[field] = dict(source_kind='user_defined', source_note=PROBE_WINDING_SOURCE_NOTE)
        if metadata or 'parameter_metadata' in part:
            part['parameter_metadata'] = metadata
        self._commit_component_operation(candidate)

    def configure_transverse_winding(self, key, *, turn_count_per_coil,
            wire_cross_section_mm2=None, insulation_thickness_mm=None,
            current_bias_a=_WINDING_INPUT_UNSET):
        """Stage a physical transverse winding and signed current biases atomically.

        The calibrated current matrix is retained read-only. Known N without
        metal area keeps bulk CAD. Clearing N and size also removes its ideal
        electrical definition, restoring legacy field admission. No current
        fitting, optical command edit or geometry enlargement is performed.
        """
        from temsim.transverse_winding import (
            AREA_FIELD, FIELDS, INSULATION_FIELD, NOTE_FIELD, STATUS_FIELD, TURN_FIELD,
            supported_transverse_winding, validate_transverse_winding,
        )
        from temsim.transverse_current import (
            BIAS_FIELD, FIELDS as CURRENT_FIELDS, MATRIX_FIELD,
            NOTE_FIELD as CURRENT_NOTE, STATUS_FIELD as CURRENT_STATUS,
            electrical_channel_count, validate_transverse_current_inputs,
        )
        candidate=deepcopy(self.document)
        by_key={part['key']:part for part in candidate['parts']}
        part=by_key.get(key)
        if (part is None or not supported_transverse_winding(part)
                or part.get('model_3d',{}).get('base',{}).get('kind','existing')!='existing'):
            raise ValueError('Select a native physical deflector or material stigmator winding')
        if turn_count_per_coil is None and (wire_cross_section_mm2 is not None or insulation_thickness_mm is not None):
            raise ValueError('Conductor and insulation size require a known per-coil turn count')
        if turn_count_per_coil is None and current_bias_a is not _WINDING_INPUT_UNSET:
            raise ValueError('Clearing a winding cannot also assign current biases')
        previous=deepcopy(part)
        metadata=deepcopy(part.get('parameter_metadata',{}))
        if turn_count_per_coil is None:
            for field in FIELDS | CURRENT_FIELDS:
                part.pop(field,None)
                metadata.pop(field,None)
        else:
            part[TURN_FIELD]=turn_count_per_coil
            part[STATUS_FIELD]='engineering_reconstruction_not_oem'
            part[NOTE_FIELD]=TRANSVERSE_WINDING_SOURCE_NOTE
            for field,value in ((AREA_FIELD,wire_cross_section_mm2),(INSULATION_FIELD,insulation_thickness_mm)):
                if value is None:
                    part.pop(field,None)
                    metadata.pop(field,None)
                else:
                    part[field]=value
            for field in (TURN_FIELD,AREA_FIELD,INSULATION_FIELD):
                if field in part and previous.get(field)!=part[field]:
                    metadata[field]=dict(source_kind='user_defined',source_note=TRANSVERSE_WINDING_SOURCE_NOTE)
            if current_bias_a is not _WINDING_INPUT_UNSET:
                if MATRIX_FIELD not in part:
                    raise ValueError('Current biases require a saved calibrated current matrix')
                old_bias=previous.get(BIAS_FIELD,[0.]*electrical_channel_count(part))
                if BIAS_FIELD in previous or current_bias_a!=old_bias:
                    part[BIAS_FIELD]=deepcopy(current_bias_a)
                if current_bias_a!=old_bias:
                    part[CURRENT_STATUS]='manual_current_edit'
                    part[CURRENT_NOTE]=('Explicit user current-bias edit in amperes. The saved current matrix '
                        'and native optical commands are retained; the previous field match is not re-fitted.')
                    metadata[BIAS_FIELD]=dict(source_kind='user_defined',source_note=part[CURRENT_NOTE])
        if metadata or 'parameter_metadata' in part:
            part['parameter_metadata']=metadata
        validate_transverse_winding(part,by_key)
        validate_transverse_current_inputs(part,by_key)
        # Applying unchanged values preserves source notes and history exactly.
        changed_fields=(FIELDS | CURRENT_FIELDS)-{NOTE_FIELD,STATUS_FIELD}
        if all(part.get(field)==previous.get(field) for field in changed_fields):
            return
        self._commit_component_operation(candidate)

    def configure_multipole_winding(self, key, *, turn_count_per_coil,
            wire_cross_section_mm2=None, insulation_thickness_mm=None,
            current_bias_a=_WINDING_INPUT_UNSET):
        """Edit fixed multipole winding inputs and signed ampere biases atomically.

        HP1/HP2 retain their existing canonical hexapole N/area fields. Auxiliary
        inputs describe an engineering winding-only reconstruction at an existing
        optical owner, preserving physical transport boundaries and commands.
        """
        from temsim.multipole_winding import (
            FIELDS, INSULATION_FIELD, NOTE_FIELD, STATUS_FIELD, area_field, turn_field,
            supported_multipole_winding, validate_multipole_winding,
        )
        from temsim.multipole_current import (
            FIELDS as CURRENT_FIELDS, MATRIX_FIELD, BIAS_FIELD,
            NOTE_FIELD as CURRENT_NOTE, STATUS_FIELD as CURRENT_STATUS,
            electrical_channel_count, validate_multipole_current_inputs,
        )
        from temsim.cad_probe_winding import (
            FIELDS as LEGACY_FIELDS, STATUS_FIELD as LEGACY_STATUS, NOTE_FIELD as LEGACY_NOTE,
        )
        from temsim.probe_hexapole_geometry import is_probe_hexapole
        candidate=deepcopy(self.document)
        by_key={part['key']:part for part in candidate['parts']}
        part=by_key.get(key)
        if (part is None or not supported_multipole_winding(part)
                or part.get('model_3d',{}).get('base',{}).get('kind','existing')!='existing'):
            raise ValueError('Select an identified native multipole winding model')
        if turn_count_per_coil is None and (wire_cross_section_mm2 is not None or insulation_thickness_mm is not None):
            raise ValueError('Conductor and insulation size require a known per-coil turn count')
        if turn_count_per_coil is None and current_bias_a is not _WINDING_INPUT_UNSET:
            raise ValueError('Clearing a winding cannot also assign current biases')
        previous=deepcopy(part)
        metadata=deepcopy(part.get('parameter_metadata',{}))
        hp=is_probe_hexapole(part)
        winding_fields=FIELDS | (LEGACY_FIELDS if hp else frozenset())
        nfield,afield=turn_field(part),area_field(part)
        if turn_count_per_coil is None:
            for field in winding_fields | CURRENT_FIELDS:
                part.pop(field,None);metadata.pop(field,None)
        else:
            part[nfield]=turn_count_per_coil
            part[STATUS_FIELD]='engineering_reconstruction_not_oem'
            part[NOTE_FIELD]=MULTIPOLE_WINDING_SOURCE_NOTE
            if hp:
                part[LEGACY_STATUS]='engineering_reconstruction_not_oem'
                part[LEGACY_NOTE]=MULTIPOLE_WINDING_SOURCE_NOTE
            for field,value in ((afield,wire_cross_section_mm2),(INSULATION_FIELD,insulation_thickness_mm)):
                if value is None:
                    part.pop(field,None);metadata.pop(field,None)
                else:part[field]=value
            for field in (nfield,afield,INSULATION_FIELD):
                if field in part and previous.get(field)!=part[field]:
                    metadata[field]=dict(source_kind='user_defined',source_note=MULTIPOLE_WINDING_SOURCE_NOTE)
            if current_bias_a is not _WINDING_INPUT_UNSET:
                if MATRIX_FIELD not in part:
                    raise ValueError('Current biases require a saved calibrated current matrix')
                old_bias=previous.get(BIAS_FIELD,[0.]*electrical_channel_count(part))
                if BIAS_FIELD in previous or current_bias_a!=old_bias:
                    part[BIAS_FIELD]=deepcopy(current_bias_a)
                if current_bias_a!=old_bias:
                    part[CURRENT_STATUS]='manual_current_edit'
                    part[CURRENT_NOTE]=('Explicit user multipole current-bias edit in amperes. The saved '
                        'matrix and native optical commands are retained; the original match is not re-fitted.')
                    metadata[BIAS_FIELD]=dict(source_kind='user_defined',source_note=part[CURRENT_NOTE])
        if metadata or 'parameter_metadata' in part:part['parameter_metadata']=metadata
        validate_multipole_winding(part,by_key)
        validate_multipole_current_inputs(part,by_key)
        notes={NOTE_FIELD,STATUS_FIELD,LEGACY_NOTE,LEGACY_STATUS}
        activating=turn_count_per_coil is not None and not FIELDS.intersection(previous)
        if not activating and all(part.get(field)==previous.get(field) for field in (winding_fields|CURRENT_FIELDS)-notes):return
        self._commit_component_operation(candidate)

    def configure_windings(self, owner_key, windings, *, ideal_coil_current_a_at_100_percent):
        """Stage a common current and one or more owned winding definitions."""
        from temsim.coil_reference_metadata import ROUND_LENS_PROFILES, OWNER_CURRENT_FIELD
        from temsim.coil_winding import WINDING_FIELDS, WIRE_AREA_FIELD, validate_winding_definition
        candidate = deepcopy(self.document)
        by_key = {part["key"]: part for part in candidate["parts"]}
        owner = by_key.get(owner_key)
        if owner is None or owner.get("mechanical_profile") not in ROUND_LENS_PROFILES:
            raise ValueError("Select an explicitly owned round-lens excitation winding")
        if not isinstance(windings, Mapping) or not windings:
            raise ValueError("Configure at least one owned winding")
        changed_turns = False
        for key, values in windings.items():
            part = by_key.get(key)
            if (part is None or part.get("mechanical_profile") != "magnetic_excitation_coil"
                    or part.get("field_source_key") != owner_key):
                raise ValueError(f"{key}: winding must belong to the selected round-lens owner")
            if (not isinstance(values, Mapping) or "winding_turn_count" not in values
                    or not set(values).issubset(WINDING_FIELDS)):
                raise ValueError(f"{key}: provide a turn count and only supported winding inputs")
            values = deepcopy(dict(values))
            if ("winding_interval_turn_counts" not in values
                    and len(part.get("winding_interval_turn_counts", ())) == 1):
                from temsim.cad_round_winding import round_winding_envelope
                if len(round_winding_envelope(part,by_key)["intervals_mm"]) == 1:
                    # One occupied interval is identical to total N, not a
                    # redistribution between independent physical windings.
                    values["winding_interval_turn_counts"] = [values["winding_turn_count"]]
            diameter_field = "winding_wire_diameter_mm"
            size_fields = {WIRE_AREA_FIELD, diameter_field}
            if size_fields.issubset(values) and all(values[name] is not None for name in size_fields):
                # Reject contradictory aliases before canonicalizing the new
                # area input. An unchanged historical dual record stays exact.
                validate_winding_definition({"winding_turn_count":values["winding_turn_count"],
                                             **{name:values[name] for name in size_fields}})
                if any(part.get(name,_WINDING_INPUT_UNSET) != values[name] for name in size_fields):
                    values[diameter_field] = None
            elif size_fields.intersection(values):
                supplied = next(name for name in size_fields if name in values and values[name] is not None) if any(
                    name in values and values[name] is not None for name in size_fields) else None
                if supplied is None:
                    values.update({name:None for name in size_fields})
                else:
                    values[next(iter(size_fields-{supplied}))] = None
            # A definition can explicitly clear optional CAD/split inputs. In
            # particular, unknown conductor size is absent rather than zero.
            for field, value in values.items():
                previous = part.get(field, _WINDING_INPUT_UNSET)
                if value is None and field != "winding_turn_count":
                    part.pop(field, None)
                else:
                    part[field] = deepcopy(value)
                if previous != part.get(field, _WINDING_INPUT_UNSET):
                    _manual_winding_source(part, field)
                    changed_turns |= field in {"winding_turn_count", "winding_interval_turn_counts"}
            validate_winding_definition(part)
        previous_current = owner.get(OWNER_CURRENT_FIELD, _WINDING_INPUT_UNSET)
        owner[OWNER_CURRENT_FIELD] = ideal_coil_current_a_at_100_percent
        if previous_current != ideal_coil_current_a_at_100_percent:
            _manual_winding_source(owner, OWNER_CURRENT_FIELD)
        elif changed_turns:
            _invalidate_current_calibration(owner, OWNER_CURRENT_FIELD)
        self._commit_component_operation(candidate)
        return owner_key

    def place_subassembly(self, key, placement):
        """Save a fixed origin or a persistent component anchor in the draft."""
        candidate = deepcopy(self.document)
        for entry in candidate.get("subassemblies", ()):
            if entry["key"] == key:
                entry["placement"] = deepcopy(placement)
                self._commit_component_operation(candidate)
                return
        raise ValueError(f"Unknown subassembly: {key}")

    def add_component(self, part):
        """Insert an independent mechanical part as one undoable operation."""
        candidate, key = added_component_document(self.document, part)
        self._commit_component_operation(candidate)
        return key

    def place_component(self, key, center_z_mm, include_children=True, *, physical_pose=None):
        """Translate module-local axial coordinates, optionally with children."""
        if key in SHARED_DEFLECTOR_HOSTS:
            raise ValueError(f"{key} uses shared hardware; place {SHARED_DEFLECTOR_HOSTS[key]} instead")
        host = bound_field_owner(self.part(key), "local_center_z_mm")
        if host:
            raise ValueError(f"{key} follows {host}; place its host or edit the coil optical reference offset")
        candidate, keys = placed_component_document(
            self.document, key, center_z_mm, include_children=include_children,
        )
        if physical_pose is not None:
            part = next(part for part in candidate["parts"] if part["key"] == key)
            self._set_physical_pose_values(part, physical_pose)
        self._commit_component_operation(candidate)
        return keys

    @classmethod
    def _set_physical_pose_values(cls, part, values):
        from temsim.lens_pose import (
            PHYSICAL_POSE_FIELDS, migrate_legacy_lens_pose,
            supports_physical_lens_pose, validate_physical_lens_pose,
        )
        if not supports_physical_lens_pose(part):
            raise ValueError("Edit a supported main-column component, or the physical host of a shared channel")
        if not isinstance(values, Mapping) or set(values) - set(PHYSICAL_POSE_FIELDS):
            raise ValueError("Choose component offsets in mm, azimuths in degrees or optical tilts in mrad")
        values = {field: cls._finite_dimension(value) for field, value in values.items()}
        migrate_legacy_lens_pose(part)
        part.update(values)
        validate_physical_lens_pose(part)

    def set_physical_pose(self, key, values):
        """Stage one main-column component pose without changing excitation."""
        part = deepcopy(self.part(key))
        self._set_physical_pose_values(part, values)
        self._commit_part(key, part)

    def copy_component_from(self, source, key, new_key, center_z_mm,
                            parent_key=None, include_children=True, *, name=None):
        """Make an independent mechanical copy without adding optical controls."""
        if key in SHARED_DEFLECTOR_HOSTS:
            raise ValueError(f"{key} is a control channel on {SHARED_DEFLECTOR_HOSTS[key]}, not independent hardware")
        candidate, new_key = copied_component_document(
            self.document, source, key, new_key, center_z_mm,
            parent_key=parent_key, include_children=include_children, name=name,
        )
        self._commit_component_operation(candidate)
        return new_key

    @staticmethod
    def _finite_dimension(value):
        try:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("A dimension must be a finite number")
            return float(value)
        except (OverflowError, TypeError) as exc:
            raise ValueError("A dimension must be a finite number") from exc

    @staticmethod
    def _numeric_target(root, path):
        if not path:
            raise ValueError("Choose an existing numeric dimension")
        current = root
        try:
            for segment in path[:-1]:
                if isinstance(current, list) and (
                    isinstance(segment, bool) or not isinstance(segment, int) or segment < 0
                ):
                    raise ValueError("Choose an existing array element")
                current = current[segment]
            final = path[-1]
            if isinstance(current, list) and (
                isinstance(final, bool) or not isinstance(final, int) or final < 0
            ):
                raise ValueError("Choose an existing array element")
            previous = current[final]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Choose an existing dimension path") from exc
        if isinstance(previous, bool) or not isinstance(previous, Real):
            raise ValueError("Choose an existing numeric dimension")
        return current, final

    @staticmethod
    def _resize_copied_axial_geometry(part, length):
        part.update(resized_copied_axial_fields(part, length))

    @staticmethod
    def _synchronize_custom_base_length(part):
        base = part.get("model_3d", {}).get("base", {})
        if (part.get("mechanical_part_role") in CUSTOM_COMPONENT_ROLES
                and base.get("kind", "existing") != "existing"):
            length = float(base["length_mm"])
            PartModelDocument._resize_copied_axial_geometry(part, length)
            part.update(resized_part_axial_coordinates(part, length))
            part["length_mm"] = length

    def set_dimension(self, path, value):
        path = tuple(path)
        value = self._finite_dimension(value)
        if len(path) < 3 or path[0] != "parts":
            raise ValueError("Only existing part dimensions can be edited")
        host = (shared_deflector_field_owner(path[1], path[2])
                or bound_field_owner(self.part(path[1]), path[2]))
        if host:
            raise ValueError(f"{path[1]} uses shared hardware; edit {host} to change its structure")
        part = deepcopy(self.part(path[1]))
        field = path[2]
        from temsim.multipole_winding import (
            FIELDS as MULTIPOLE_FIELDS, INSULATION_FIELD as MULTIPOLE_INSULATION,
            area_field,turn_field,
        )
        from temsim.multipole_current import BIAS_FIELD as MULTIPOLE_BIAS, MATRIX_FIELD as MULTIPOLE_MATRIX
        if field==MULTIPOLE_MATRIX or field.startswith('ideal_multipole_calibration_reference_'):
            raise ValueError('The saved multipole calibration reference is read-only; edit signed current biases in Configure winding')
        multipole_inputs={turn_field(part),area_field(part),MULTIPOLE_INSULATION,MULTIPOLE_BIAS}
        if field in multipole_inputs and (MULTIPOLE_MATRIX in part or MULTIPOLE_FIELDS.intersection(part)):
            if field not in part:
                raise ValueError('Configure this multipole winding input before editing its dimension')
            nfield,afield=turn_field(part),area_field(part)
            if field==nfield and (value<=0 or not value.is_integer()):
                raise ValueError('Per-coil turn count must be a positive integer')
            biases=_WINDING_INPUT_UNSET
            if field==MULTIPOLE_BIAS:
                if len(path)!=4 or not isinstance(path[3],int):
                    raise ValueError('Select one existing multipole current-bias channel')
                biases=deepcopy(part[field]);biases[path[3]]=value
            elif len(path)!=3:
                raise ValueError('Select one multipole winding scalar input')
            self.configure_multipole_winding(path[1],
                turn_count_per_coil=int(value) if field==nfield else part[nfield],
                wire_cross_section_mm2=value if field==afield else part.get(afield),
                insulation_thickness_mm=value if field==MULTIPOLE_INSULATION else part.get(MULTIPOLE_INSULATION),
                current_bias_a=biases)
            return
        from temsim.transverse_winding import (
            AREA_FIELD as TRANSVERSE_AREA, INSULATION_FIELD as TRANSVERSE_INSULATION,
            TURN_FIELD as TRANSVERSE_TURNS,
        )
        from temsim.transverse_current import BIAS_FIELD as TRANSVERSE_BIAS
        from temsim.transverse_current import MATRIX_FIELD as TRANSVERSE_MATRIX, REFERENCE_ENERGY_FIELD as TRANSVERSE_REFERENCE
        if field in {TRANSVERSE_MATRIX,TRANSVERSE_REFERENCE}:
            raise ValueError('The saved current calibration reference is read-only; edit signed current biases in Configure winding')
        if field in {TRANSVERSE_AREA,TRANSVERSE_INSULATION,TRANSVERSE_TURNS,TRANSVERSE_BIAS}:
            if field not in part:
                raise ValueError('Configure this transverse winding input before editing its dimension')
            if field==TRANSVERSE_TURNS and (value<=0 or not value.is_integer()):
                raise ValueError('Per-coil turn count must be a positive integer')
            biases=_WINDING_INPUT_UNSET
            if field==TRANSVERSE_BIAS:
                if len(path)!=4 or not isinstance(path[3],int):
                    raise ValueError('Select one existing current-bias channel')
                biases=deepcopy(part[field])
                biases[path[3]]=value
            elif len(path)!=3:
                raise ValueError('Select one transverse winding scalar input')
            self.configure_transverse_winding(path[1],
                turn_count_per_coil=int(value) if field==TRANSVERSE_TURNS else part[TRANSVERSE_TURNS],
                wire_cross_section_mm2=value if field==TRANSVERSE_AREA else part.get(TRANSVERSE_AREA),
                insulation_thickness_mm=value if field==TRANSVERSE_INSULATION else part.get(TRANSVERSE_INSULATION),
                current_bias_a=biases)
            return
        from temsim.cad_probe_winding import AREA_FIELD, TURN_FIELD
        if len(path) == 3 and field in {AREA_FIELD, TURN_FIELD}:
            if field not in part:
                raise ValueError('Configure this radial winding input before editing its dimensions')
            if field == TURN_FIELD and (value <= 0 or not value.is_integer()):
                raise ValueError('Per-coil turn count must be a positive integer')
            self.configure_probe_winding(path[1],
                turn_count_per_coil=int(value) if field == TURN_FIELD else part[TURN_FIELD],
                wire_cross_section_mm2=value if field == AREA_FIELD else part.get(AREA_FIELD))
            return
        if len(path) == 3 and field == "winding_turn_count":
            if field not in part or value <= 0 or not value.is_integer():
                raise ValueError("Choose an existing positive integer conductor turn count")
            turns = int(value)
            if part[field] == turns:
                return
            candidate = deepcopy(self.document)
            by_key = {p['key']:p for p in candidate['parts']}
            coil = by_key[path[1]]
            coil[field] = turns
            _manual_winding_source(coil,field)
            if len(coil.get('winding_interval_turn_counts',())) == 1:
                from temsim.cad_round_winding import round_winding_envelope
                if len(round_winding_envelope(coil,by_key)['intervals_mm']) == 1:
                    coil['winding_interval_turn_counts'] = [turns]
                    _manual_winding_source(coil,'winding_interval_turn_counts')
            owner = by_key.get(coil.get('field_source_key'),{})
            current_field = 'ideal_coil_current_a_at_100_percent'
            if current_field in owner:
                _invalidate_current_calibration(owner,current_field)
            self._commit_component_operation(candidate)
            return
        if len(path) == 3 and field == "winding_wire_cross_section_mm2":
            if field not in part:
                raise ValueError("Configure a conductor metal cross-section before editing its dimension")
            if part[field] == value:
                return
            part[field] = value
            part.pop("winding_wire_diameter_mm",None)
            _manual_winding_source(part,field)
            _manual_winding_source(part,"winding_wire_diameter_mm")
            self._commit_part(path[1],part)
            return
        from temsim.dimension_edit_policy import dimension_reference_reason
        if reason := dimension_reference_reason(part, field):
            raise ValueError(reason)
        from temsim.lens_pose import PHYSICAL_POSE_FIELDS, supports_physical_lens_pose
        if len(path) == 3 and field in PHYSICAL_POSE_FIELDS and supports_physical_lens_pose(part):
            self.set_physical_pose(path[1], {field: value})
            return
        from temsim.part_model_apertures import has_optional_plate_thickness, validate_optional_plate_thickness
        if len(path) == 3 and field == "plate_thickness_mm" and has_optional_plate_thickness(part):
            part[field] = value
            validate_optional_plate_thickness(part)
            self._commit_part(path[1], part)
            return
        if field == "model_3d":
            from temsim.part_model_features import validate_model_3d

            current, final = self._numeric_target(part, path[2:])
            current[final] = value
            validate_model_3d(part)
            if path[2:] == ("model_3d", "base", "length_mm"):
                self._synchronize_custom_base_length(part)
            self._commit_part(path[1], part)
            return
        if field not in part or not field.endswith(("_mm", "_um", "_nm", "_deg", "_mrad")):
            raise ValueError("Choose an existing dimension in mm, µm, nm, degrees or mrad")
        from temsim.optics.electron_gun.tip_assembly import is_tip_part
        if is_tip_part(part) and len(path) == 3:
            if isinstance(part[field], bool) or not isinstance(part[field], Real):
                raise ValueError("Choose a numeric dimension")
            part[field] = value
            self._commit_part(path[1], part)
            return
        if field == "local_center_z_mm":
            raise ValueError("The axial centre is fixed; edit the existing length or endpoints")
        if len(path) == 3:
            if isinstance(part[field], bool) or not isinstance(part[field], Real):
                raise ValueError("Choose a numeric dimension")
            if field == "length_mm":
                endpoints = resized_part_axial_coordinates(part, value)
                self._resize_copied_axial_geometry(part, value)
                part.update(endpoints)
                base = part.get("model_3d", {}).get("base", {})
                if (part.get("mechanical_part_role") in CUSTOM_COMPONENT_ROLES
                        and base.get("kind", "existing") != "existing"):
                    base["length_mm"] = value
            part[field] = float(value)
            if field in {"local_start_z_mm", "local_end_z_mm"}:
                part["length_mm"] = float(part["local_end_z_mm"]) - float(part["local_start_z_mm"])
        else:
            current, index = self._numeric_target(part[field], path[3:])
            current[index] = float(value)
            if field == "material_intervals_mm":
                from temsim.magnetic_geometry import corrector_return_extent
                part.update(corrector_return_extent(part))
        self._commit_part(path[1], part)

    def set_model_3d(self, key, configuration):
        """Replace one mechanical CAD configuration, or remove it with None.

        Schema errors are rejected atomically. A schema-valid boolean operation
        may remain a draft until complete geometry is validated on Save.
        Existing physical dimensions and solver parameters are left unchanged.
        """
        from temsim.part_model_features import validate_model_3d

        host = bound_field_owner(self.part(key), "model_3d")
        if host:
            raise ValueError(f"{key} is a control channel; edit {host} to change its structure")
        candidate = deepcopy(self.part(key))
        if configuration is None:
            candidate.pop("model_3d", None)
        elif isinstance(configuration, Mapping):
            candidate["model_3d"] = deepcopy(dict(configuration))
        else:
            raise ValueError("model_3d must be a table or None")
        validate_model_3d(candidate)
        self._synchronize_custom_base_length(candidate)
        self._commit_part(key, candidate)

    def update_model_3d(self, key, updates):
        """Merge nested tables in one undo step; replace supplied collections."""
        from temsim.part_model_features import default_model_3d

        if not isinstance(updates, Mapping):
            raise ValueError("model_3d updates must be a table")
        part = self.part(key)
        configuration = deepcopy(part.get("model_3d", default_model_3d(part)))

        def merge(target, values):
            for field, value in values.items():
                if isinstance(value, Mapping) and isinstance(target.get(field), dict):
                    merge(target[field], value)
                else:
                    target[field] = deepcopy(value)

        merge(configuration, updates)
        self.set_model_3d(key, configuration)

    def upsert_model_feature(self, key, feature):
        """Add or replace a stable feature id without changing other features."""
        from temsim.part_model_features import default_model_3d

        if not isinstance(feature, Mapping) or not isinstance(feature.get("id"), str) or not feature["id"]:
            raise ValueError("A feature needs a non-empty stable id")
        part = self.part(key)
        configuration = deepcopy(part.get("model_3d", default_model_3d(part)))
        features = configuration.setdefault("features", [])
        for index, previous in enumerate(features):
            if previous.get("id") == feature["id"]:
                features[index] = deepcopy(dict(feature))
                break
        else:
            features.append(deepcopy(dict(feature)))
        self.set_model_3d(key, configuration)

    def remove_model_feature(self, key, feature_id):
        configuration = deepcopy(self.part(key).get("model_3d", {"schema_version": 1}))
        features = configuration.get("features", [])
        if not any(feature.get("id") == feature_id for feature in features):
            raise ValueError(f"Unknown model feature: {feature_id}")
        configuration["features"] = [feature for feature in features if feature.get("id") != feature_id]
        self.set_model_3d(key, configuration)

    def assign_material(self, key, material_key, region="body"):
        from temsim.part_materials import part_material_updates
        updates = part_material_updates(self.part(key), material_key, region=region)
        for path, value in updates.items():
            if tuple(path[:2]) != ("parts", key) or len(path) != 3:
                raise ValueError("Invalid material assignment path")
            self.part(key)[path[2]] = deepcopy(value)
        self._remember()

    def undo(self):
        if self.can_undo:
            self._history_index -= 1
            self.document = deepcopy(self._history[self._history_index])

    def redo(self):
        if self.can_redo:
            self._history_index += 1
            self.document = deepcopy(self._history[self._history_index])

    def revert(self):
        self.document = deepcopy(self._baseline)
        self._history = [deepcopy(self.document)]
        self._history_index = 0

    def updates(self):
        from temsim.coil_winding import WINDING_FIELDS
        from temsim.cad_probe_winding import FIELDS as PROBE_WINDING_FIELDS
        from temsim.transverse_winding import FIELDS as TRANSVERSE_WINDING_FIELDS
        from temsim.transverse_current import FIELDS as TRANSVERSE_CURRENT_FIELDS
        from temsim.multipole_winding import FIELDS as MULTIPOLE_WINDING_FIELDS
        from temsim.multipole_current import FIELDS as MULTIPOLE_CURRENT_FIELDS
        originals = {part["key"]: part for part in self._baseline["parts"]}
        updates = {}
        additions = []
        for part in self.document["parts"]:
            if part["key"] not in originals:
                additions.append(deepcopy(part))
                continue
            original = originals[part["key"]]
            for field, value in part.items():
                if bound_field_owner(part, field):
                    # Persist the authoritative host edit only. Derived
                    # coordinates are resolved again on load, never competing
                    # writes to the control channel or its anchored insert.
                    continue
                if field not in original or value != original[field]:
                    updates[("parts", part["key"], field)] = deepcopy(value)
            if "material_regions" in original and "material_regions" not in part:
                updates[("parts", part["key"], "material_regions")] = {}
            if "model_3d" in original and "model_3d" not in part:
                updates[("parts", part["key"], "model_3d")] = None
            for field in ((WINDING_FIELDS - {"winding_turn_count"}) | PROBE_WINDING_FIELDS
                    | TRANSVERSE_WINDING_FIELDS | TRANSVERSE_CURRENT_FIELDS
                    | MULTIPOLE_WINDING_FIELDS | MULTIPOLE_CURRENT_FIELDS):
                if field in original and field not in part:
                    updates[("parts", part["key"], field)] = None
        current_keys = {part["key"] for part in self.document["parts"]}
        removed = tuple(key for key in originals if key not in current_keys)
        previous = {entry["key"]: entry["placement"] for entry in self._baseline.get("subassemblies", ())}
        placements = {entry["key"]: deepcopy(entry["placement"])
                      for entry in self.document.get("subassemblies", ())
                      if previous.get(entry["key"]) != entry["placement"]}
        return PartChangeSet(updates, tuple(additions), removed, self._source_bytes,
                             dict(self._dependency_bytes), placements, True)

    def reload(self):
        """Read an external source revision only after this draft is resolved."""
        if self.dirty:
            raise ValueError("Save a copy or Revert the current draft before reloading")
        candidate = type(self)(self.path)
        self._source_bytes = candidate._source_bytes
        self._dependency_bytes = candidate._dependency_bytes
        self._subassembly_owners = candidate._subassembly_owners
        self._editable_text = candidate._editable_text
        self._baseline = candidate._baseline
        self.document = candidate.document
        self._history = candidate._history
        self._history_index = 0

    def validate(self):
        validate_component_graph(self.document)
        module_manifest.validate_document(self.document)

    def assert_source_current(self):
        if self._archive_identity is not None:
            raise ValueError("Archived structure is read-only; load a live assembly before saving structural edits")
        if getattr(self, "_new_document", False):
            if self.path.exists():
                raise ValueError("The new destination now exists; Save copy to avoid replacing it")
            return
        if self.path.read_bytes() != self._source_bytes:
            raise ValueError("The source file changed outside this editor. Save a copy or reopen it before saving.")
        from temsim.shared_tip import dependencies
        if dependencies(self.path) != self._dependency_bytes:
            raise ValueError("An assembly dependency changed outside this editor. Save a copy or reopen it before saving.")

    def save(self, *, project_save=None):
        self.assert_source_current()
        self.validate()
        if getattr(self, "_new_document", False):
            staged = self.independent_copy_text()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation protects a new definition from a concurrent
            # file appearing after the destination was chosen in the dialog.
            with self.path.open("x", encoding="utf-8", newline="") as stream:
                stream.write(staged)
            self._accept_saved_file(self.path, expected=tomllib.loads(staged))
            return
        updates = self.updates()
        if not updates:
            return
        staged = module_manifest.stage_manifest_text(
            self._editable_text, updates
        )
        # Schema, Boolean and catalog preparation may take time. Recheck the
        # source before either persistence route can write the staged edit.
        self.assert_source_current()
        if project_save is not None:
            if project_save(self.path, updates) is False:
                raise ValueError("The project did not save the model draft")
        else:
            from temsim.shared_tip import catalog_root_for, catalog_definitions
            root = catalog_root_for(self.path)
            if root is not None and (self._dependency_bytes or self.path in catalog_definitions(root)):
                from temsim.component_persistence import save_component_changes
                import os
                save_component_changes(root, os.path.relpath(self.path, root), updates, None)
            else:
                if self.document.get("subassemblies"):
                    raise ValueError("Save a composed assembly inside its catalog, or use Save copy for an independent file")
                module_manifest._atomic_write_text(self.path, staged)
        self._accept_saved_file(self.path, expected=tomllib.loads(staged))

    def save_copy(self, path):
        destination = Path(path).resolve()
        if destination == self.path:
            raise ValueError("Use Save for the current file")
        from temsim.shared_tip import catalog_root_for, catalog_definitions
        root = catalog_root_for(destination)
        if root is not None and destination in catalog_definitions(root):
            raise ValueError("Use Save to edit a linked physical definition; choose an independent destination for a copy")
        staged = self.independent_copy_text()
        module_manifest._atomic_write_text(destination, staged)
        self._accept_saved_file(destination, expected=tomllib.loads(staged))

    def independent_copy_text(self):
        """Validated, materialized candidate definition without filesystem edits."""
        self.validate()
        staged = module_manifest.stage_manifest_text(
            self._editable_text, self.updates()
        )
        # Save copy is explicitly independent, including a module with shared
        # sources. Resolve its values before removing the source link.
        parsed = tomllib.loads(staged)
        if parsed.get("subassemblies") or any("tip_definition_file" in part for part in parsed["parts"]):
            import tomli_w
            parsed.pop("subassemblies", None)
            for part in parsed["parts"]:
                part.pop("tip_definition_file", None)
            staged = tomli_w.dumps(parsed)
        return staged

    def _accept_saved_file(self, path, *, expected):
        source = path.read_bytes()
        from temsim.shared_tip import materialized_text
        parsed = tomllib.loads(materialized_text(source.decode("utf-8-sig"), path))
        if parsed != expected:
            raise ValueError("The saved file does not match the draft. The draft is retained; inspect the source before saving again.")
        self.path = path
        self._new_document = False
        self._source_bytes = source
        from temsim.shared_tip import materialized_text, dependencies
        self._dependency_bytes = dependencies(path)
        self._editable_text = materialized_text(source.decode("utf-8-sig"), path)
        self._baseline = resolve_field_hardware_parts(tomllib.loads(self._editable_text))
        self.document = deepcopy(self._baseline)
        from temsim.subassemblies import ownership
        self._subassembly_owners = ownership(self._baseline, self.path)
        self._history[self._history_index] = deepcopy(self.document)

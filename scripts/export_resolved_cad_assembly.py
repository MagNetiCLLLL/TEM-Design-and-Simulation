"""Make one globally placed SolidSpec catalog from frozen assembly-audit input.

Only the standard library is imported. Branch mounts and component poses have
already been applied by the producer; this adapter adds module translations
once. The resulting JSON is consumed by scripts/export_cad_models.py in the
optional CAD environment. Catalog creation itself is not a collision pass.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re


DEFAULT_ASSEMBLY = "FEG | None | C3 + Probe Corrector + Image Corrector | No Energy Filter"
IDENTITY = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _label(display, identity):
    # Human-readable prefix plus an identity-derived suffix. The full original
    # strings are retained in metadata; unsafe separators never become paths.
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", str(display)).strip("._")[:48] or "part"
    return prefix + "--" + _digest(identity)[:16]


def _offset(placement):
    if "rotation" in placement or "rotation_matrix" in placement or "transform" in placement:
        raise ValueError("Only prepared translations are accepted; poses and branch rotation are already in each shape")
    value = placement.get("offset_mm", [0., 0., 0.])
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("Module offset requires three finite millimetre coordinates")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError("Module offset requires three finite millimetre coordinates")
    return [float(x) for x in value]


def resolved_catalog(document, *, assembly_key=DEFAULT_ASSEMBLY,
                     source_path=None, source_file_sha256=None):
    """Resolve one captured assembly without loading current files or defaults."""
    if document.get("schema") != "temsim-material-collision-input-v1" or document.get("units") != "mm":
        raise ValueError("Expected frozen temsim-material-collision-input-v1 in millimetres")
    candidates = [a for a in document["assemblies"] if a["key"] == assembly_key]
    if len(candidates) != 1:
        raise ValueError(f"Choose one unique captured assembly key: {assembly_key!r}")
    assembly = candidates[0]
    if assembly.get("material_liner_scope") != "physical_adapter":
        raise ValueError("Complete assembly export requires the declared physical liner adapter")
    modules = {m["key"]: m for m in document["modules"]}
    if len(modules) != len(document["modules"]):
        raise ValueError("Captured module keys must be unique")
    if not any(p["module_key"].startswith("@physical_liners:") for p in assembly["placements"]):
        raise ValueError("Complete assembly export requires an actual physical liner placement")

    solids, units, omitted, placement_records = [], [], [], []
    identities, exported_ids, unit_names, placement_names = set(), set(), set(), set()
    for placement in assembly["placements"]:
        module_key = placement["module_key"]
        if module_key not in modules:
            raise ValueError(f"Missing captured module {module_key}")
        module = modules[module_key]
        namespace = placement.get("instance_key", module_key)
        if not isinstance(namespace, str) or not namespace or namespace in placement_names:
            raise ValueError("Repeated module placement requires a unique nonempty instance_key")
        placement_names.add(namespace)
        offset = _offset(placement)
        is_liner = module_key.startswith("@physical_liners:")
        if is_liner and any(offset):
            raise ValueError("Physical liner adapter shapes are already global; their placement must be zero")
        aliases, local_rows = {}, []
        for original in module["solids"]:
            spec = deepcopy(original)
            key, region = original["key"], original["region"]
            if not all(isinstance(v, str) and v for v in (key, region)):
                raise ValueError("Material key and region must be nonempty strings")
            local_id = original.get("instance_key", key + "/" + region)
            if not isinstance(local_id, str) or not local_id:
                raise ValueError("Solid instance_key must be a nonempty string")
            identity = (namespace, local_id)
            if identity in identities:
                raise ValueError(f"Duplicate material instance: {identity}")
            identities.add(identity)
            # Liner segments can share key/region. Never collapse them into a
            # single leaf or silently discard a repeated physical tube.
            part_identity = (namespace, key, original.get("instance_key"))
            spec["key"] = _label(key, part_identity)
            spec["region"] = _label(region, (region,))
            global_id = spec["key"] + "/" + spec["region"]
            if global_id in exported_ids:
                raise ValueError("Generated CAD identity collision")
            exported_ids.add(global_id)
            spec["shape"] = ({"kind": "transform", "shape": spec["shape"],
                "rotation": deepcopy(IDENTITY), "offset_mm": offset.copy()}
                if any(offset) else spec["shape"])
            spec["resolved_instance"] = dict(module_key=module_key,
                module_instance_key=namespace, part_key=key, region=region,
                source_instance_key=original.get("instance_key"),
                generated_liner_instance_key=local_id if is_liner else None,
                source_shape_sha256=_digest(original["shape"]), module_offset_mm=offset.copy(),
                placement_application_count=1 if any(offset) else 0,
                coordinates="global_column_mm")
            solids.append(spec)
            row = dict(id=global_id, original_key=key, local_id=local_id,
                       instance_key=original.get("instance_key"))
            local_rows.append(row)
            for alias in {local_id, key + "/" + region}:
                aliases.setdefault(alias, []).append(global_id)

        assigned = set()

        def add_unit(source_key, name, ids, source_scope):
            unit_key = _label(source_key, (namespace, source_scope, source_key))
            if unit_key in unit_names:
                raise ValueError("Generated CAD unit identity collision")
            unit_names.add(unit_key)
            units.append(dict(key=unit_key, name=f"{module.get('name', module_key)}: {name}",
                solid_ids=ids, source_module_key=module_key, source_module_instance_key=namespace,
                source_unit_key=source_key, ownership_scope=source_scope))

        for unit in module.get("units", ()):
            ids = []
            for reference in unit["solid_ids"]:
                choices = aliases.get(reference, ())
                if len(choices) != 1:
                    raise ValueError(f"Unit reference {reference!r} is missing or ambiguous in {module_key}")
                if choices[0] in assigned:
                    raise ValueError("A material instance may belong to only one exported unit")
                assigned.add(choices[0])
                ids.append(choices[0])
            if not ids:
                raise ValueError("An exported unit must contain material")
            add_unit(unit["key"], unit.get("name", unit["key"]), ids, "captured_unit")
        unassigned = {}
        for row in local_rows:
            if row["id"] not in assigned:
                owner = row["instance_key"] or row["original_key"]
                unassigned.setdefault(owner, []).append(row["id"])
        for owner, ids in unassigned.items():
            add_unit(owner, owner, ids, "physical_liner_instance" if is_liner else "captured_part")
        omitted.extend(dict(source_module_key=module_key, source_module_instance_key=namespace,
                            record=deepcopy(row)) for row in module.get("omitted", ()))
        placement_records.append(dict(module_key=module_key, module_instance_key=namespace,
            source=module.get("source"), offset_mm=offset,
            input_coordinates="global_column_mm" if is_liner else "posed_module_local_mm",
            branch_rotation_reapplied=False, solids=len(local_rows),
            diagnostic_geometry_override=deepcopy(module.get("diagnostic_geometry_override"))))
    if not solids:
        raise ValueError("Selected assembly contains no material solids")
    return dict(schema="temsim-analytic-material-cad-v1", units="mm",
        coordinate_system="global_column_mm",
        scope="Captured generic mechanical reconstruction; not OEM CAD. Export preparation does not establish collision clearance or optical qualification.",
        provenance=dict(source_input=source_path, source_file_sha256=source_file_sha256,
            source_document_sha256=_digest(document), selected_assembly_key=assembly_key,
            selection=deepcopy(assembly["selection"]), capture_scope=document.get("capture_scope"),
            source_files=deepcopy(document.get("source_files", [])),
            producer_implementation_sha256=deepcopy(document.get("implementation_sha256", {})),
            adapter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            placements=placement_records, material_liner_scope=assembly["material_liner_scope"],
            liner_material_decisions=deepcopy(assembly.get("liner_material_decisions", [])),
            assembly_scope=assembly.get("scope"),
            placement_rule="Outer module translation exactly once; preserve all existing component and branch transforms; physical liners are already global"),
        modules=[dict(key="resolved_column_assembly", name=assembly_key,
            source=source_path, solids=solids, units=units, omitted=omitted,
            coordinate_system="global_column_mm")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, help="Frozen inputs.json from validate_cad_assemblies.py prepare")
    parser.add_argument("--output", type=Path, required=True, help="Global SolidSpec catalog JSON")
    parser.add_argument("--assembly-key", default=DEFAULT_ASSEMBLY, help="Exact captured assembly key")
    parser.add_argument("--list-assemblies", action="store_true", help="List captured keys without exporting")
    args = parser.parse_args()
    data = args.inputs.read_bytes()
    document = json.loads(data.decode("utf-8-sig"))
    if args.list_assemblies:
        print("\n".join(row["key"] for row in document["assemblies"]))
        return
    value = resolved_catalog(document, assembly_key=args.assembly_key,
        source_path=str(args.inputs.resolve()), source_file_sha256=hashlib.sha256(data).hexdigest())
    if args.output.resolve() == args.inputs.resolve():
        raise ValueError("Output must not replace the frozen input")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(catalog=str(args.output.resolve()), assembly=args.assembly_key,
        solid_count=len(value["modules"][0]["solids"]), unit_count=len(value["modules"][0]["units"]),
        material_liner_scope=value["provenance"]["material_liner_scope"])))


if __name__ == "__main__":
    main()

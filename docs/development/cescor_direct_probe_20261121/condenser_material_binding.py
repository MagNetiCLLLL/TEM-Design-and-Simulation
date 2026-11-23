"""Bind the inherited material solutions to the current optical assembly before using a gun handoff."""

import hashlib
import json
import sys

import numpy as np

from condenser_source_fields import EV, FEM, FINAL, HERE, MAT
from temsim.part_materials import material_catalog
from temsim.physics.lens_field_provider import lens_geometry_binding


def check_materials(state):
    sys.path.insert(0, str(HERE.parent / "probe_circuit_audit_20261122"))
    from circuit_audit import audit_inputs

    live, _, _ = audit_inputs()
    with np.load(MAT) as raw, np.load(FEM) as table:
        if json.loads(json.dumps(live)) != json.loads(str(raw["recipe"])):
            raise ValueError(
                "The joint magnetic solution differs from the installed geometry or B-H assignments"
            )
        solved = json.loads(str(raw["report"]))
        report = json.loads(str(table["report"]))
        if (
            solved["currents_A"] != report["currents_A"]
            or report["currents_A"]["probe_tl12_lens"] != FINAL["TL12_A"]
            or solved["mesh_id"] != report["mesh_id"]
        ):
            raise ValueError(
                "The registered corrector table has a different operating current or mesh"
            )
    obj = json.loads((EV / "joint-objective-material-field.json").read_text())
    binding = lens_geometry_binding(state, "objective_lens")
    geometry_id = hashlib.sha256(binding.canonical_geometry_json.encode()).hexdigest()
    iron = next(
        x["bh_material"] for x in material_catalog() if x["material_key"] == "armco_pure_iron"
    )
    if (
        geometry_id != obj["installed_geometry_digest"]
        or iron["source_sha256"] != obj["curve_sha256"]
        or abs(FINAL["objective_At"]) != obj["ampere_turns"]
    ):
        raise ValueError("The objective current, material curve or pole geometry has changed")

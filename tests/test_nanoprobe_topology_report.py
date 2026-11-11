"""Optical assignment of finite regions from synthetic executed checkpoints."""
import json
from types import SimpleNamespace

import numpy as np

from scripts.compare_nanoprobe_topology import SCHEMA, _audit_cohort, compare_reports


def _report(focus_mm, planes):
    lenses = [("C2", .5), ("C3", 2.5)]
    boundaries = [("start", 0.), *lenses, ("end", 4.)]
    tx, ty = np.meshgrid(np.linspace(-1., 1., 32), np.linspace(-1., 1., 32))
    tx, ty = tx.ravel() * 1e-3, ty.ravel() * 1e-3
    drift = (planes - focus_mm)[:, None] * 1e-3
    cp = SimpleNamespace(z_mm=planes, x_m=drift * tx, y_m=drift * ty,
                         tx_rad=np.broadcast_to(tx, (len(planes), len(tx))),
                         ty_rad=np.broadcast_to(ty, (len(planes), len(ty))))
    masks = np.ones(cp.x_m.shape, bool)
    cohort = _audit_cohort(cp, masks, np.ones(len(tx)), np.arange(len(tx)),
                           boundaries, lenses, 3.5, ())
    return {"schema": SCHEMA, "round_lenses": lenses, "component_planes": boundaries,
            "z_bounds_mm": [0., 4.], "cohorts": {"actual": cohort}}


def test_nearest_lens_changes_inside_the_same_component_interval():
    planes = np.linspace(0., 4., 81)
    reference = _report(1.21, planes)
    same_relation = _report(1.31, planes)
    other_lens = _report(1.71, planes)

    assert compare_reports(reference, same_relation)["status"] == "PRESERVED_AT_BOUNDED_SAMPLING"
    assert compare_reports(json.loads(json.dumps(reference)), same_relation)["status"] == "PRESERVED_AT_BOUNDED_SAMPLING"
    comparison = compare_reports(reference, other_lens)
    assert comparison["status"] == "CHANGED"
    actual = comparison["cohorts"]["actual"]
    assert actual["reference"][0]["component_interval"] == actual["candidate"][0]["component_interval"]
    assert actual["reference"][0]["lens_keys"] == ["C2"]
    assert actual["candidate"][0]["lens_keys"] == ["C3"]


def test_bracket_crossing_nearest_lens_boundary_stays_unresolved():
    planes = np.linspace(0., 4., 80)
    reference = _report(1.21, planes)
    ambiguous = _report(1.49, planes)

    assert compare_reports(reference, ambiguous)["status"] == "UNRESOLVED"
    assert ambiguous["cohorts"]["actual"]["unresolved"][0]["reason"] == "nearest_lens_relation_not_resolved"

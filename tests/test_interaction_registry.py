"""Ownership contracts independent of any propagation implementation."""

from dataclasses import FrozenInstanceError, replace
import json

import pytest

from temsim.physics.interaction_registry import (
    InteractionOwnershipError, InteractionRecord, InteractionRegistry,
)


def record(name, *, owner="c2", family="magnetic", role="field",
           interval=(100., 120.), coverage=("c2/magnetic",), **kwargs):
    return InteractionRecord(name, owner, family, role, interval,
                             covered_effects=coverage, source_digest="frozen-field-a", **kwargs)


def test_registry_json_roundtrip_is_detached_immutable_and_complete():
    drives = ["c2_y", "c2_x"]
    original = record("c2-field", circuit_id="c1-c2", logical_drive_ids=drives)
    source = [original]
    registry = InteractionRegistry(source)
    source.clear()
    drives.clear()
    payload = json.loads(json.dumps(registry.to_dict(), allow_nan=False))
    restored = InteractionRegistry.from_dict(payload)
    assert restored == registry
    assert restored.digest == registry.digest
    assert restored.validate() is restored
    assert restored.records[0].validate() is restored.records[0]
    assert restored.records[0].logical_drive_ids == ("c2_x", "c2_y")
    assert payload["records"][0]["interval_unit"] == "mm"
    payload["records"][0]["interval_mm"][0] = -900.
    assert restored.records[0].interval_mm == (100., 120.)
    with pytest.raises(FrozenInstanceError):
        restored.records[0].owner_id = "somewhere-else"
    with pytest.raises(FrozenInstanceError):
        restored.records = ()


def test_distinct_sources_and_multipole_families_can_overlap_one_host():
    registry = InteractionRegistry((
        record("round", owner="shared-host", coverage=("round/focus",)),
        record("a1", owner="shared-host", family="quadrupole", coverage=("a1/field",)),
        record("a2", owner="shared-host", family="hexapole", coverage=("a2/field",)),
        record("neighbor", owner="c3", coverage=("c3/magnetic",)),
    ))
    assert registry.executing_action_ids == ("round", "a1", "a2", "neighbor")


def test_legacy_additional_cc_and_cs_need_distinct_owned_responses():
    field = record("native-focusing")
    cs = record("cs", family="spherical", role="residual", interval=(110., 110.),
                coverage=("c2/legacy-additional-cs",), residual_provenance="legacy-compatible explicit lens Cs")
    cc = record("cc", family="chromatic", role="residual", interval=(110., 110.),
                coverage=("c2/legacy-additional-cc",), residual_provenance="legacy-compatible explicit additional Cc")
    registry = InteractionRegistry((field, cs, cc))
    assert registry.executing_action_ids == ("native-focusing", "cs", "cc")


def test_residual_label_does_not_authorize_reapplying_a_field_total():
    total = record("mapped-lens", coverage=("c2/magnetic", "c2/total-cs", "c2/total-cc"))
    duplicate = record("copied-total-cc", role="residual", family="chromatic",
                       interval=(110., 110.), coverage=("c2/total-cc",),
                       residual_provenance="copied field-derived total")
    with pytest.raises(InteractionOwnershipError, match="c2/total-cc.*executed twice"):
        InteractionRegistry((total, duplicate))


@pytest.mark.parametrize("member", ("c1/magnetic", "c2/magnetic"))
def test_nonlinear_circuit_total_cannot_be_reexecuted_as_members(member):
    total = record("joint-total", owner="c1-c2-circuit", circuit_id="c1-c2",
                   coverage=("c1/magnetic", "c2/magnetic"),
                   logical_drive_ids=("c1-current", "c2-current"))
    local = record("duplicate-member", owner=member.split("/")[0],
                   coverage=(member,), circuit_id="c1-c2")
    with pytest.raises(InteractionOwnershipError, match="executed twice"):
        InteractionRegistry((total, local))


@pytest.mark.parametrize("overlap_role", ("field", "residual"))
def test_complete_reduced_map_replaces_fields_and_explicit_actions(overlap_role):
    reduced = record("image-map", owner="image-interval", role="reduced",
                     coverage=("objective/focus", "p1/focus", "p1/additional-cs"))
    direct = record("duplicate-direct", owner="p1", role=overlap_role,
                    interval=(110., 110.) if overlap_role == "residual" else (105., 125.),
                    coverage=("p1/additional-cs" if overlap_role == "residual" else "p1/focus",),
                    **({"residual_provenance": "legacy explicit Cs"} if overlap_role == "residual" else {}))
    with pytest.raises(InteractionOwnershipError, match="image-map.*duplicate-direct"):
        InteractionRegistry((reduced, direct))


def test_observations_and_field_projections_are_not_executed_again():
    field = record("field")
    observed = record("field-view", role="observation", action_id="field")
    fitted = record("measured-cs", role="observation", family="measured-spherical")
    registry = InteractionRegistry((field, observed, fitted))
    assert registry.executing_action_ids == ("field",)
    assert not observed.executes
    assert InteractionRegistry((observed, fitted)).executing_action_ids == ()


@pytest.mark.parametrize("role", ("field", "geometry_boundary", "material", "detector"))
def test_exact_physical_duplicates_are_rejected_even_with_different_record_ids(role):
    original = record("first", role=role)
    with pytest.raises(InteractionOwnershipError, match="executed twice"):
        InteractionRegistry((original, replace(original, interaction_id="second", action_id="second")))
    with pytest.raises(InteractionOwnershipError, match="Duplicate interaction ID"):
        InteractionRegistry((original, original))


def test_adjacent_distributed_intervals_are_owned_once():
    left = record("left", interval=(100., 110.))
    right = record("right", interval=(110., 120.))
    registry = InteractionRegistry((left, right))
    assert registry.executing_action_ids == ("left", "right")


def test_point_action_conflicts_at_both_endpoints_of_a_covered_response():
    left = record("left", interval=(100., 110.))
    point = record("point", interval=(110., 110.))
    with pytest.raises(InteractionOwnershipError, match="executed twice"):
        InteractionRegistry((left, point))
    with pytest.raises(InteractionOwnershipError, match="executed twice"):
        InteractionRegistry((record("right", interval=(110., 120.)), point))
    with pytest.raises(InteractionOwnershipError, match="executed twice"):
        InteractionRegistry((point, record("other-point", interval=(110., 110.))))
    outside = record("outside", interval=(110.001, 110.001))
    assert InteractionRegistry((left, outside)).executing_action_ids == ("left", "outside")


def test_symmetric_half_steps_count_as_one_declared_physical_action():
    left = record("pre", action_id="distributed-field", execution_fraction=.5)
    right = record("post", action_id="distributed-field", execution_fraction=.5)
    registry = InteractionRegistry((left, right))
    assert registry.executing_action_ids == ("distributed-field",)
    assert InteractionRegistry.from_dict(json.loads(json.dumps(registry.to_dict()))) == registry
    with pytest.raises(InteractionOwnershipError, match="executed twice"):
        InteractionRegistry((left, right, record("duplicate-complete-field")))


@pytest.mark.parametrize("fractions", ((.5,), (.5, .25), (1., 1.)))
def test_incomplete_or_repeated_split_actions_are_not_silently_admitted(fractions):
    pieces = tuple(record(str(i), action_id="one-field", execution_fraction=fraction)
                   for i, fraction in enumerate(fractions))
    with pytest.raises(InteractionOwnershipError, match="partitions must sum to one"):
        InteractionRegistry(pieces)


@pytest.mark.parametrize("change", (
    {"owner_id": "other"}, {"covered_effects": ("other/field",)},
    {"source_digest": "different-field"}, {"interval_mm": (101., 120.)},
    {"representation": "reduced"},
))
def test_one_action_id_cannot_hide_incompatible_split_sources(change):
    first = record("pre", action_id="same", execution_fraction=.5)
    second = replace(first, interaction_id="post", **change)
    with pytest.raises(InteractionOwnershipError, match="incompatible numerical partitions"):
        InteractionRegistry((first, second))


@pytest.mark.parametrize("bad_interval", (
    (2., 1.), (float("nan"), 1.), (0., float("inf")),
    (False, 1.), ("0", 1.), (0.,), (0., 1., 2.), (0, 10**400),
))
def test_invalid_intervals_fail_with_explicit_metadata_errors(bad_interval):
    with pytest.raises(InteractionOwnershipError, match="interval_mm"):
        record("invalid", interval=bad_interval)


@pytest.mark.parametrize("change", (
    {"interaction_id": ""}, {"owner_id": " x"}, {"effect_family": "x\n"},
    {"representation": "implicit_guess"}, {"covered_effects": "c2/magnetic"},
    {"covered_effects": ("same", "same")}, {"logical_drive_ids": ("x", "x")},
    {"source_digest": 2}, {"circuit_id": ""}, {"action_id": ""},
    {"execution_fraction": 0.}, {"execution_fraction": -1.},
    {"execution_fraction": float("nan")}, {"execution_fraction": True},
))
def test_invalid_identity_role_and_split_metadata_is_rejected(change):
    with pytest.raises(InteractionOwnershipError):
        replace(record("good"), **change)


@pytest.mark.parametrize("provenance", (None, "", " ", 12))
def test_residual_requires_explicit_nonempty_provenance(provenance):
    with pytest.raises(InteractionOwnershipError, match="residual_provenance"):
        record("residual", role="residual", residual_provenance=provenance)


def test_residual_provenance_cannot_be_assigned_to_a_field_record():
    with pytest.raises(InteractionOwnershipError, match="requires the residual"):
        record("field", residual_provenance="ambiguous")


@pytest.mark.parametrize("mutation", (
    lambda d: d.update(schema="unknown"),
    lambda d: d.update(interval_unit="m"),
    lambda d: d.update(unknown_coefficient=3),
    lambda d: d.pop("covered_effects"),
))
def test_standalone_record_json_requires_explicit_known_contract(mutation):
    data = record("field").to_dict()
    mutation(data)
    with pytest.raises(InteractionOwnershipError):
        InteractionRecord.from_dict(data)


def test_registry_json_revalidates_ownership_and_rejects_unknown_metadata():
    data = InteractionRegistry((record("field"),)).to_dict()
    data["records"].append(record("duplicate").to_dict())
    with pytest.raises(InteractionOwnershipError, match="executed twice"):
        InteractionRegistry.from_dict(data)
    for invalid in ({"schema": "unknown", "records": []},
                    {"schema": data["schema"], "records": [], "extra": False},
                    {"schema": data["schema"], "records": "not-an-array"}):
        with pytest.raises(InteractionOwnershipError):
            InteractionRegistry.from_dict(invalid)


def test_digest_covers_scientific_ownership_metadata():
    initial = record("field", circuit_id="circuit", logical_drive_ids=("drive",))
    baseline = InteractionRegistry((initial,)).digest
    for changes in ({"interval_mm": (100., 121.)}, {"source_digest": "field-b"},
                    {"owner_id": "other"}, {"covered_effects": ("different/response",)},
                    {"logical_drive_ids": ("other-drive",)}, {"circuit_id": "other-circuit"}):
        assert InteractionRegistry((replace(initial, **changes),)).digest != baseline


def test_coverage_and_drive_order_do_not_change_effect_identity():
    first = record("field", coverage=("b", "a"), logical_drive_ids=("y", "x"))
    second = record("field", coverage=("a", "b"), logical_drive_ids=("x", "y"))
    assert InteractionRegistry((first,)).digest == InteractionRegistry((second,)).digest


def test_empty_registry_and_default_atomic_coverage_have_unambiguous_contracts():
    assert InteractionRegistry.from_dict(InteractionRegistry().to_dict()) == InteractionRegistry()
    value = InteractionRecord("action", "lens", "focus", "field", (0., 1.))
    assert value.covered_effects == ("lens/focus",)
    assert value.action_id == "action"

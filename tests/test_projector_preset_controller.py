"""Image projection solves off-thread without rebuilding the incident optics."""
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.gui import operating_preset_controller as module
from temsim.immutable_json import json_digest
from temsim.instrument_snapshot import encode_instrument
from temsim.optics.column import default_state


@pytest.mark.parametrize("cancel", [False, True])
def test_projector_worker_preserves_captured_graph_and_cancels_atomically(qtbot, monkeypatch, cancel):
    state = default_state()
    state.objective_lens.percent = 54.3
    state.objective_lens.cs_mm = 3.21
    state.lens_intrinsic_aberrations = {"condenser_lens_2": {"a1_mm": .031}}
    state.electron_gun._projector_test_source_parameter = .123
    original = json_digest(encode_instrument(state))
    catalog = AssemblyCatalog()
    entered, release = Event(), Event()
    observed = []

    def solve(captured, key, **kwargs):
        callback = captured._tuning_cancelled
        del captured._tuning_cancelled
        observed.append((json_digest(encode_instrument(captured)), get_ident()))
        captured._tuning_cancelled = callback
        captured.projector_mode = "image"
        entered.set()
        assert release.wait(10.)
        return SimpleNamespace(summary="test projection only", changed_devices=())

    def forbidden(*args, **kwargs):
        raise AssertionError("Projection-only worker must not rebuild layout or apply condenser settings")

    monkeypatch.setattr(module, "apply_projector_mode", solve)
    monkeypatch.setattr(module, "apply_operating_mode_pair", forbidden)
    monkeypatch.setattr(module, "apply_physical_layout_to_state", forbidden)
    controller = module.OperatingPresetController()
    delivered, errors = [], []
    controller.result_ready.connect(lambda *args: delivered.append(args))
    controller.failed.connect(errors.append)
    controller.submit(state, catalog, catalog.default_selection(), "nano_probe", "imaging",
                      projector_only=True)
    try:
        qtbot.waitUntil(lambda: entered.is_set() or bool(errors), timeout=10_000)
        assert not errors
        assert observed[0][0] == original
        assert observed[0][1] != get_ident()
        assert json_digest(encode_instrument(state)) == original
        if cancel:
            controller.invalidate_pending()
    finally:
        release.set()
        assert controller.pool.waitForDone(10_000)
    qtbot.wait(30)
    assert not errors
    assert json_digest(encode_instrument(state)) == original
    if cancel:
        assert not delivered
    else:
        assert len(delivered) == 1
        candidate = delivered[0][0]
        assert candidate.projector_mode == "image"
        assert candidate.objective_lens.cs_mm == 3.21
        assert candidate.electron_gun._projector_test_source_parameter == .123
        assert not hasattr(candidate, "_tuning_cancelled")


def test_invalid_projector_only_request_does_not_cancel_pending_work(monkeypatch):
    controller = module.OperatingPresetController()
    invalidations = []
    monkeypatch.setattr(controller, "invalidate_pending", lambda: invalidations.append(True))
    with pytest.raises(ValueError, match="current assembly"):
        controller.submit(None, None, None, "nano_probe", "imaging",
                          load_assembly=True, projector_only=True)
    assert not invalidations


def test_projector_worker_reads_captured_inputs_before_catalog_and_solve(qtbot, monkeypatch, tmp_path):
    """Exact State cloning must retain the model-input owner of its worker."""
    from hashlib import sha256
    from temsim import input_io
    state = default_state()
    path = tmp_path/"captured-input.txt"
    path.write_text("new live input", encoding="utf-8")
    captured = b"original captured input"
    archive = dict(schema=input_io.ARCHIVE_SCHEMA, config_root=str(tmp_path),
                   runtime=input_io.runtime_identity(), files=[dict(
                       path=str(path), config_relative="captured-input.txt",
                       content_hex=captured.hex(), sha256=sha256(captured).hexdigest())])
    archive["digest"] = json_digest(archive)
    input_io.bind_archive(state, archive)
    graph = encode_instrument(state)
    selection = AssemblyCatalog().default_selection()
    observed = []

    def compatible(family, *args):
        observed.append(("catalog", input_io.read_text(path)))
        return [SimpleNamespace(key="imaging" if family == "projector" else "nano_probe")]

    def solve(candidate, *args, **kwargs):
        assert input_io.archive_payload(candidate) == input_io.archive_payload(state)
        observed.append(("solve", input_io.read_text(path)))
        return SimpleNamespace(summary="archive ownership probe", changed_devices=())

    monkeypatch.setattr(module, "compatible_modes", compatible)
    monkeypatch.setattr(module, "apply_projector_mode", solve)
    worker = module._Worker(0, type(state), graph, AssemblyCatalog(), selection,
                            "nano_probe", "imaging", False, projector_only=True)
    errors = []
    worker.signals.error.connect(errors.append)
    worker.run()
    assert not errors
    assert observed == [("catalog", "original captured input"),
                        ("catalog", "original captured input"),
                        ("solve", "original captured input")]
    assert json_digest(encode_instrument(state)) == json_digest(graph)
    assert input_io.active_archive() is None


@pytest.mark.parametrize("load_assembly", [False, True])
@pytest.mark.parametrize("long_body", [False, True])
def test_preset_worker_preserves_width_overrides_in_selected_catalog(
        qtbot, monkeypatch, tmp_path, load_assembly, long_body):
    from temsim.part_model_document import PartModelDocument
    from temsim.shared_tip import copy_catalog_tree

    reference = AssemblyCatalog()
    root = tmp_path / "instruments"
    copy_catalog_tree(reference.root, root)
    defaults = {"condenser_stigmator": 4.5, "objective_stigmator": 4.5,
                "diffraction_stigmator": 6.0}
    for relative, keys in (
        ("column/C3_ProbeCorrector.toml", ("condenser_stigmator", "objective_stigmator")),
        ("project_and_recording_system/NoEnergyFilter.toml", ("diffraction_stigmator",)),
    ):
        draft = PartModelDocument(root / relative)
        for key in keys:
            draft.set_dimension(("parts", key, "effective_length_mm"), defaults[key])
        if long_body and "condenser_stigmator" in keys:
            draft.set_dimension(("parts", "condenser_stigmator", "length_mm"), 40.0)
        draft.save()
    catalog = AssemblyCatalog(root)
    selection = catalog.default_selection()
    state = default_state()
    catalog.apply(state, selection)
    overrides = {"condenser_stigmator": 35.0 if long_body else 2.0,
                 "objective_stigmator": 2.5, "diffraction_stigmator": 3.0}
    for item in state.stigmators:
        item.length_mm = overrides[item.key]
        item.strength_x_percent, item.strength_y_percent = 4.0, -2.0
    before = json_digest(encode_instrument(state))
    observed = []

    def solve(snapshot, *_args, **_kwargs):
        assert snapshot._resolved_assembly.root == root.resolve()
        observed.append({item.key: item.length_mm for item in snapshot.stigmators})
        return None

    # Isolate preparation from the separate preset calibration algorithm.
    monkeypatch.setattr(module, "apply_operating_mode_pair", solve)
    controller = module.OperatingPresetController()
    errors = []
    controller.failed.connect(errors.append)
    with qtbot.waitSignal(controller.result_ready, timeout=30_000) as delivered:
        controller.submit(state, catalog, selection, "nano_probe", "diffraction",
                          load_assembly=load_assembly)
    assert controller.pool.waitForDone(30_000)
    assert not errors
    expected = defaults if load_assembly else overrides
    assert observed == [expected]
    assert {item.key: item.length_mm for item in delivered.args[0].stigmators} == expected
    assert json_digest(encode_instrument(state)) == before

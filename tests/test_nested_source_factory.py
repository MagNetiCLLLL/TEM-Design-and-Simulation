"""Default-source and historical-input configuration; no particle transport."""
from pathlib import Path
import glob
import tomllib

from temsim.optics.electron_gun.emitter import HALTON_PREFIX_SAMPLING_SCHEMA
from temsim.optics.electron_gun.field_emission import (
    _create_emitter, _component_payload, _restore_component_settings,
)


def test_new_planar_factory_explicitly_selects_the_nested_source():
    emitter = _create_emitter()
    assert emitter.surface_model is None and emitter.coherence is None
    assert emitter.sampling.schema == HALTON_PREFIX_SAMPLING_SCHEMA
    assert emitter.sampling.seed == 0
    emitter.sampling.validate_count(3000)


def test_explicit_legacy_component_does_not_acquire_a_new_sampling_identity():
    emitter = _create_emitter()
    record = _component_payload(emitter)
    record.pop("sampling", None)
    restored = _create_emitter()
    _restore_component_settings(restored, record)
    assert restored.sampling is None


def test_explicit_nested_component_keeps_its_schema_through_input_restore():
    emitter = _create_emitter()
    record = _component_payload(emitter)
    restored = _create_emitter()
    restored.sampling = None
    _restore_component_settings(restored, record)
    assert restored.sampling == emitter.sampling


def test_wheel_literal_data_inputs_exist_in_the_checkout():
    root = Path(__file__).resolve().parents[1]
    specification = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    for inputs in specification["tool"]["setuptools"]["data-files"].values():
        for value in inputs:
            if not glob.has_magic(value):
                assert (root / value).is_file(), value

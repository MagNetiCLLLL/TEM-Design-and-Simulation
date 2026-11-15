"""Derivative-kernel changes must invalidate numerical field identities."""
from pathlib import Path

from temsim.magnetic_field_scene import _implementation_inputs
from temsim.physics import compiled_annular_coil
from temsim.physics.frozen_copper_coil_field import FrozenCopperCoilField


def test_compiled_axis_source_bytes_are_part_of_field_implementation(monkeypatch):
    # File revisions are loaded by a fresh interpreter; do not reuse the
    # running interpreter's intentionally memoized implementation inventory.
    original = _implementation_inputs.__wrapped__(FrozenCopperCoilField)
    key = "physics/compiled_annular_coil.py"
    assert key in original
    target = Path(compiled_annular_coil.__file__).resolve()
    read_bytes = Path.read_bytes

    def changed_source(path):
        content = read_bytes(path)
        return content+b"\n# diagnostic implementation change\n" if path.resolve() == target else content

    monkeypatch.setattr(Path, "read_bytes", changed_source)
    changed = _implementation_inputs.__wrapped__(FrozenCopperCoilField)
    assert changed[key] != original[key]
    assert {name: value for name, value in changed.items() if name != key} == {
        name: value for name, value in original.items() if name != key}


def test_compiler_arithmetic_and_optional_version_are_explicit():
    metadata = compiled_annular_coil.annular_axis_execution_metadata()
    assert metadata["fastmath"] is False
    assert "numba_version" in metadata
    assert type(metadata["compiled_available"]) is bool

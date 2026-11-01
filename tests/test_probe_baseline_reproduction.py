from hashlib import sha256
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_solver_verification_accepts_only_recorded_newline_conversion(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from reproduce_probe_baseline import verify_solver_sources

    code = b"coefficient = 1\r\n"
    manifest = {"files": {"solver.py": {"raw_sha256": sha256(code).hexdigest(),
                                       "lf_sha256": sha256(code.replace(b"\r\n", b"\n")).hexdigest()}}}
    file = tmp_path / "solver.py"
    file.write_bytes(code)
    assert verify_solver_sources(manifest, tmp_path) == []
    file.write_bytes(code.replace(b"\r\n", b"\n"))
    assert verify_solver_sources(manifest, tmp_path) == ["solver.py"]
    file.write_bytes(b"coefficient = 2\n")
    with pytest.raises(ValueError, match="code changed"):
        verify_solver_sources(manifest, tmp_path)
    file.write_bytes(code)
    (tmp_path / "extra.py").write_bytes(b"")
    with pytest.raises(ValueError, match="inventory"):
        verify_solver_sources(manifest, tmp_path)


def test_source_bound_geometry_is_compared_by_actual_stops(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from audit_probe_drives import unchanged_interception_geometry

    original = dict(schema="test", scope="vacuum", start_z_m=0., stop_z_m=1.,
                    boundaries=[{"bore": .001}], aperture_inventory=[], digest="old",
                    source_scene_geometry_digest="old-source")
    after = dict(original, digest="new", source_scene_geometry_digest="new-source")
    wrap = lambda data: SimpleNamespace(to_dict=lambda: data)
    assert unchanged_interception_geometry(wrap(original), wrap(after))
    after["boundaries"] = [{"bore": .0009}]
    assert not unchanged_interception_geometry(wrap(original), wrap(after))


def test_sampled_field_rank_distinguishes_dependent_drives(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from audit_probe_drives import normalized_rank

    assert normalized_rank([np.array([1., 0.]), np.array([0., 1000.])])["rank"] == 2
    assert normalized_rank([np.array([1., 0.]), np.array([1000., 0.])])["rank"] == 1
    with pytest.raises(ValueError, match="no field response"):
        normalized_rank([np.zeros(2)])


def test_field_contract_ignores_only_code_receipts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from reproduce_probe_baseline import field_content_digest

    old = dict(numerical_digest="old", scene={"aperture": 1},
               contributions=[dict(source_id="instrument_electric", potential=dict(
                   digest="old", arrays={"phi": [0., 1.]}, metadata=dict(
                       request={"implementation_sha256": {"solver": "old"}, "voltage": 300000.},
                       source_identities={"request": "old", "numerical": "old", "geometry": "fixed"})))])
    new = deepcopy(old)
    new["numerical_digest"] = "new"
    new["contributions"][0]["potential"]["metadata"]["request"]["implementation_sha256"]["solver"] = "new"
    assert field_content_digest(old) == field_content_digest(new)
    new["contributions"][0]["potential"]["arrays"]["phi"][1] = 1.1
    assert field_content_digest(old) != field_content_digest(new)

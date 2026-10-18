"""Operating preset caches must follow the captured model-input owner."""
from hashlib import sha256
import re
import tomllib
from types import SimpleNamespace

import pytest

from temsim import input_io
from temsim.immutable_json import json_digest
from temsim.operating_modes import load_operating_mode_catalog, mode_by_key
from temsim.paths import CONFIG_ROOT, OPERATING_MODE_CONFIG_ROOT


def _archived_catalog(original, percent):
    # Modify only an in-memory copy of the actual TOML catalog. The first P2
    # entry belongs to Image; parsing below verifies the intended edit.
    text = original.decode("utf-8")
    text, count = re.subn(r"projector_lens_2 = \{ percent = [^}]+\}",
                         f"projector_lens_2 = {{ percent = {percent} }}", text, count=1)
    assert count == 1
    imaging = next(row for row in tomllib.loads(text)["modes"] if row["key"] == "imaging")
    assert imaging["devices"]["projector_lens_2"]["percent"] == percent
    content = text.encode("utf-8")
    path = (OPERATING_MODE_CONFIG_ROOT/"catalog.toml").absolute()
    archive = dict(schema=input_io.ARCHIVE_SCHEMA, config_root=str(CONFIG_ROOT.absolute()),
                   runtime=input_io.runtime_identity(), files=[dict(
                       path=str(path), config_relative="operating_modes/catalog.toml",
                       content_hex=content.hex(), sha256=sha256(content).hexdigest())])
    archive["digest"] = json_digest(archive)
    owner = SimpleNamespace()
    input_io.bind_archive(owner, archive)
    return owner


@pytest.mark.parametrize("archive_first", [False, True])
def test_catalog_cache_isolates_live_and_distinct_captured_presets(archive_first):
    path = OPERATING_MODE_CONFIG_ROOT/"catalog.toml"
    original = path.read_bytes()
    live_row = next(row for row in tomllib.loads(original.decode("utf-8"))["modes"]
                    if row["key"] == "imaging")
    live_percent = live_row["devices"]["projector_lens_2"]["percent"]
    first, second = _archived_catalog(original, 23.4567), _archived_catalog(original, 27.6543)
    requests = [(None, live_percent), (first, 23.4567), (second, 27.6543), (None, live_percent)]
    if archive_first:
        requests.insert(0, (first, 23.4567))
    load_operating_mode_catalog.cache_clear()
    try:
        for owner, expected in requests:
            with input_io.input_scope(owner, inherit=False):
                catalog = load_operating_mode_catalog()
                assert mode_by_key("imaging", catalog).devices["projector_lens_2"]["percent"] == expected
                assert load_operating_mode_catalog() is catalog
        assert input_io.active_archive() is None
    finally:
        load_operating_mode_catalog.cache_clear()
        assert path.read_bytes() == original

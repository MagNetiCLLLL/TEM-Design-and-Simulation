"""Checked numeric storage for append-only gun transport provenance."""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

SCHEMA = "gun-prefix-transport-extensions-v1"
FIELDS = ("emission_phase", "incremental_source", "transport_partitions")


def pack_gun_extensions(trace, arrays):
    """Retain native partitions using the existing closed section codec."""
    values = {name: getattr(trace, name) for name in FIELDS}
    if values["emission_phase"] is None and values["incremental_source"] is None and not values["transport_partitions"]:
        return None
    from temsim.particle_section_io import _data_classes, _pack
    # Reuse already stored arrays by identity without replacing their values.
    memo = {id(value): key for key, value in arrays.items() if isinstance(value, np.ndarray)}
    return {"schema": SCHEMA, "graph": _pack(values, arrays, memo, _data_classes())}


def unpack_gun_extensions(record, arrays):
    """Read known data only; historical traces have no incremental seed."""
    if record is None:
        return {}
    if not isinstance(record, Mapping) or set(record) != {"schema", "graph"} or record["schema"] != SCHEMA:
        raise ValueError("Unknown gun prefix transport extension record")
    from temsim.particle_section_io import _data_classes, _unpack
    from temsim.optics.electron_gun.base import GunTraceResult
    values = _unpack(record["graph"], arrays, _data_classes())
    if not isinstance(values, dict) or set(values) != set(FIELDS):
        raise ValueError("Invalid gun prefix transport extension fields")
    if any(values[name] is not None and not isinstance(values[name], Mapping)
           for name in ("emission_phase", "incremental_source")):
        raise ValueError("Gun source provenance requires explicit mappings")
    parts = values["transport_partitions"]
    if not isinstance(parts, tuple) or any(type(part) is not GunTraceResult for part in parts):
        raise ValueError("Gun transport partitions require original gun traces")
    return values

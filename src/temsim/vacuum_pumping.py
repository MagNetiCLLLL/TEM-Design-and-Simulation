"""Editable vacuum-service drawings, independent of electron transport.

These connections describe installed services and illustrative valve states.
They do not solve pressure, conductance, pump performance or interlocks.
"""
from __future__ import annotations

from copy import deepcopy


PUMPING_SCHEMA = "vacuum-pumping-layout-v1"
PUMP_KINDS = ("igp", "turbo", "scroll", "rotary", "cryo", "diffusion", "other")
LINE_ROLES = ("high_vacuum", "roughing", "backing", "transfer")


def _text(value, description, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{description} must be {'a string' if empty else 'a nonempty string'}")
    return value


def _rows(layout, key, fields):
    rows = layout.get(key)
    if not isinstance(rows, list):
        raise ValueError(f"Pumping {key} must be a list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError(f"Each pumping {key} entry requires exactly: {', '.join(sorted(fields))}")
        identity = _text(row["key"], f"Pumping {key} key")
        if ":" in identity or identity != identity.strip():
            raise ValueError("Pumping keys must not contain ':' or surrounding whitespace")
        if identity in seen:
            raise ValueError(f"Duplicate pumping {key} key: {identity}")
        seen.add(identity)
    return rows


def validate_pumping_layout(layout, region_keys):
    """Return a detached valid drawing or raise ValueError.

    ``{}`` means a legacy map using the default schematic. An explicit schema
    with three empty lists deliberately draws no pumping network.
    """
    if not isinstance(layout, dict):
        raise ValueError("Pumping layout must be a table")
    if not layout:
        return {}
    if set(layout) != {"schema", "pumps", "auxiliaries", "lines"}:
        raise ValueError("Pumping layout requires schema, pumps, auxiliaries and lines")
    if layout["schema"] != PUMPING_SCHEMA:
        raise ValueError(f"Unknown pumping layout schema: {layout['schema']}")
    pumps = _rows(layout, "pumps", {"key", "name", "kind"})
    auxiliaries = _rows(layout, "auxiliaries", {"key", "name", "anchor"})
    lines = _rows(layout, "lines", {"key", "source", "target", "role", "valve", "open"})
    endpoints = {f"region:{key}" for key in region_keys}
    for pump in pumps:
        _text(pump["name"], "Pump name")
        if pump["kind"] not in PUMP_KINDS:
            raise ValueError(f"Unknown pump kind: {pump['kind']}")
        endpoints.add(f"pump:{pump['key']}")
    for auxiliary in auxiliaries:
        _text(auxiliary["name"], "Auxiliary chamber name")
        _text(auxiliary["anchor"], "Auxiliary chamber display anchor")
        endpoints.add(f"aux:{auxiliary['key']}")
    for line in lines:
        for field in ("source", "target"):
            endpoint = _text(line[field], f"Line {field}")
            if endpoint not in endpoints:
                raise ValueError(f"Line {line['key']} references missing endpoint {endpoint}")
        if line["source"] == line["target"]:
            raise ValueError(f"Line {line['key']} cannot connect an endpoint to itself")
        if line["role"] not in LINE_ROLES:
            raise ValueError(f"Unknown pumping line role: {line['role']}")
        _text(line["valve"], "Valve label", empty=True)
        if type(line["open"]) is not bool:
            raise ValueError("Pumping valve open state must be boolean")
        if not line["valve"].strip() and not line["open"]:
            raise ValueError("A line without a valve is always open")
    return deepcopy(layout)


def default_pumping_layout(regions):
    """Illustrative shared services for known beam-path region keys.

    Unknown user regions receive no invented pump assignment. Airlock is an
    off-axis service chamber anchored visually to the specimen, never a new
    electron-transport interval.
    """
    keys = {region if isinstance(region, str) else region.key for region in regions}
    layout = {
        "schema": PUMPING_SCHEMA,
        "pumps": [
            {"key": "igp_tip", "name": "Gun IGP", "kind": "igp"},
            {"key": "igp_accelerator", "name": "Accelerator IGP", "kind": "igp"},
            {"key": "igp_column", "name": "Column IGP", "kind": "igp"},
            {"key": "igp_post_column", "name": "Post-specimen IGP", "kind": "igp"},
            {"key": "turbo", "name": "Shared turbo", "kind": "turbo"},
            {"key": "scroll", "name": "Dry backing / roughing", "kind": "scroll"},
        ],
        "auxiliaries": [
            {"key": "airlock", "name": "Holder airlock", "anchor": "sample"},
        ],
        "lines": [],
    }

    def line(key, source, target, role, valve="", opened=True):
        layout["lines"].append({
            "key": key, "source": source, "target": target,
            "role": role, "valve": valve, "open": opened,
        })

    for region, pump in (
        ("gun_tip", "igp_tip"),
        ("gun_accelerator", "igp_accelerator"),
        ("column", "igp_column"),
        ("specimen", "igp_column"),
        ("post_column", "igp_post_column"),
    ):
        if region in keys:
            line(f"{region}_igp", f"region:{region}", f"pump:{pump}", "high_vacuum")
    if "projection" in keys:
        line("projection_turbo", "region:projection", "pump:turbo",
             "high_vacuum", "Projection isolation", True)
    line("airlock_turbo", "aux:airlock", "pump:turbo",
         "high_vacuum", "Airlock high vacuum", False)
    line("turbo_backing", "pump:turbo", "pump:scroll", "backing")
    line("airlock_roughing", "aux:airlock", "pump:scroll",
         "roughing", "Airlock roughing", False)
    if "specimen" in keys:
        line("airlock_transfer", "aux:airlock", "region:specimen",
             "transfer", "Holder transfer isolation", False)
    return validate_pumping_layout(layout, keys)


def effective_pumping_layout(vacuum_map):
    """Resolve legacy defaults without modifying the map or enabling vacuum."""
    pumping = getattr(vacuum_map, "pumping", {})
    if isinstance(pumping, dict) and not pumping:
        return default_pumping_layout(vacuum_map.regions)
    return validate_pumping_layout(pumping, (region.key for region in vacuum_map.regions))

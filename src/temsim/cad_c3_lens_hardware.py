"""Compatibility facade for the configured C3 insulated end-plate family."""

from temsim.cad_round_lens_end_hardware import (
    STATUS, _number, end_hardware_fields, lens_end_hardware_solids,
    validate_lens_end_hardware,
)

PREFIX = "mechanical_c3_end_"
FIELDS = end_hardware_fields(PREFIX)


def validate_c3_lens_hardware(row, by_key=None):
    return validate_lens_end_hardware(row, by_key, prefix=PREFIX, label="C3")


def c3_lens_hardware_solids(part, by_key=None):
    return lens_end_hardware_solids(part, by_key, prefix=PREFIX, label="C3")

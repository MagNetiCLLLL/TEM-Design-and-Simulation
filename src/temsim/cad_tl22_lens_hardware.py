"""Saved TL22 end hardware, sized from its own transfer-lens winding."""

from temsim.cad_round_lens_end_hardware import (
    STATUS, end_hardware_fields, lens_end_hardware_solids, validate_lens_end_hardware,
)

PREFIX = "mechanical_tl22_end_"
FIELDS = end_hardware_fields(PREFIX)


def validate_tl22_lens_hardware(row, by_key=None):
    return validate_lens_end_hardware(row, by_key, prefix=PREFIX, label="TL22")


def tl22_lens_hardware_solids(part, by_key=None):
    return lens_end_hardware_solids(part, by_key, prefix=PREFIX, label="TL22")

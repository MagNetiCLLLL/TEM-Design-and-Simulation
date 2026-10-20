"""Configured Adapter Lens hardware using the shared round-lens construction."""

from temsim.cad_round_lens_end_hardware import (
    STATUS, end_hardware_fields, lens_end_hardware_solids, validate_lens_end_hardware,
)

PREFIX = "mechanical_adapter_end_"
FIELDS = end_hardware_fields(PREFIX)


def validate_adapter_lens_hardware(row, by_key=None):
    return validate_lens_end_hardware(row, by_key, prefix=PREFIX, label="Adapter Lens")


def adapter_lens_hardware_solids(part, by_key=None):
    return lens_end_hardware_solids(part, by_key, prefix=PREFIX, label="Adapter Lens")

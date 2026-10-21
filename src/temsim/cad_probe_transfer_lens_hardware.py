"""Explicit passive end hardware for the lower probe transfer lenses.

TL21 and TL1 (stored as TL12) use their own saved winding interfaces. The
construction family is shared with the other round lenses; no dimensions,
currents or optical settings are inferred from a transfer-lens name.
"""

from temsim.cad_round_lens_end_hardware import (
    STATUS, end_hardware_fields, lens_end_hardware_solids, validate_lens_end_hardware,
)

PREFIX = "mechanical_transfer_end_"
FIELDS = end_hardware_fields(PREFIX)


def validate_probe_transfer_lens_hardware(row, by_key=None):
    return validate_lens_end_hardware(row, by_key, prefix=PREFIX, label="Probe transfer lens")


def probe_transfer_lens_hardware_solids(part, by_key=None):
    return lens_end_hardware_solids(part, by_key, prefix=PREFIX, label="Probe transfer lens")

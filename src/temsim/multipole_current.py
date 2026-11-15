"""Fixed electrical references for ideal prescribed-current corrector multipoles.

Matrices map the existing signed native normal/skew strengths to physical
amperes. They are calibration inputs, not functions of the electron energy.
"""
from collections.abc import Mapping
from numbers import Integral, Real
import math
import re


PREFIX = 'ideal_multipole_'
MATRIX_FIELD = PREFIX + 'current_matrix_a_per_unit'
BIAS_FIELD = PREFIX + 'current_bias_a'
REFERENCE_ENERGY_FIELD = PREFIX + 'calibration_reference_energy_ev'
REFERENCE_ORIENTATION_FIELD = PREFIX + 'calibration_reference_orientation_rad'
KIND_FIELD = PREFIX + 'calibration_kind'
STATUS_FIELD = PREFIX + 'calibration_status'
NOTE_FIELD = PREFIX + 'calibration_source_note'
FIELDS = frozenset((MATRIX_FIELD, BIAS_FIELD, REFERENCE_ENERGY_FIELD,
    REFERENCE_ORIENTATION_FIELD, KIND_FIELD, STATUS_FIELD, NOTE_FIELD))
CALIBRATION_KIND = 'native_peak_multipole_coefficient_at_reference_plane_v1'


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must contain finite real numbers')
    return float(value)


def electrical_channel_count(part):
    from temsim.multipole_winding import supported_multipole_winding
    if not supported_multipole_winding(part):
        raise ValueError('Multipole electrical inputs require an identified winding family')
    key = str(part.get('geometry_template_key', part.get('key', '')))
    return 1 if re.fullmatch(r'(?:probe|image)_hp[12]_hexapole', key) else 2


def validate_multipole_current_inputs(part, by_key=None):
    """Return a detached reference, or None for unchanged native legacy inputs."""
    if not isinstance(part, Mapping):
        raise ValueError('Multipole electrical inputs require a component mapping')
    supplied = {str(key) for key in part if str(key).startswith(PREFIX)}
    if supplied - FIELDS:
        raise ValueError(f'{part.get("key")}: unknown ideal multipole electrical inputs: {sorted(supplied-FIELDS)}')
    if not supplied:
        return None
    if part.get('model_3d', {}).get('base', {}).get('kind', 'existing') != 'existing':
        return None
    required = FIELDS - {BIAS_FIELD, REFERENCE_ORIENTATION_FIELD}
    if not required.issubset(part):
        raise ValueError('A multipole current reference requires matrix, energy, kind, status and source note')
    from temsim.multipole_winding import multipole_winding_definition
    definition = multipole_winding_definition(part, by_key)
    if definition is None:
        raise ValueError('Multipole physical current requires an explicit winding recipe')
    turns = definition['turn_count_per_coil']
    if isinstance(turns, bool) or not isinstance(turns, Integral) or turns <= 0:
        raise ValueError('Multipole physical currents require a positive per-coil turn count')
    count = int(definition['electrical_channel_count'])
    if count not in (1, 2):
        raise ValueError('Multipole electrical source requires one or two channels')
    matrix = part[MATRIX_FIELD]
    if not isinstance(matrix, (list, tuple)) or len(matrix) != count or any(
            not isinstance(row, (list, tuple)) or len(row) != count for row in matrix):
        raise ValueError(f'Multipole current matrix must have shape {count} x {count}')
    matrix = [[_number(value, MATRIX_FIELD) for value in row] for row in matrix]
    bias = part.get(BIAS_FIELD, [0.] * count)
    if not isinstance(bias, (list, tuple)) or len(bias) != count:
        raise ValueError(f'Multipole current bias must contain {count} signed amperes')
    bias = [_number(value, BIAS_FIELD) for value in bias]
    energy = _number(part[REFERENCE_ENERGY_FIELD], REFERENCE_ENERGY_FIELD)
    if energy <= 0:
        raise ValueError('Multipole calibration reference energy must be positive')
    if part[KIND_FIELD] != CALIBRATION_KIND:
        raise ValueError('Unsupported multipole current calibration kind')
    if any(not isinstance(part[name], str) or not part[name].strip() for name in (STATUS_FIELD, NOTE_FIELD)):
        raise ValueError('Multipole calibration requires meaningful status and provenance')
    orientation = (None if REFERENCE_ORIENTATION_FIELD not in part else
                   _number(part[REFERENCE_ORIENTATION_FIELD], REFERENCE_ORIENTATION_FIELD))
    if count == 1 and orientation is None:
        raise ValueError('A single-basis hexapole requires its fixed native orientation reference')
    return dict(matrix_a_per_unit=matrix, current_bias_a=bias,
        reference_energy_ev=energy, reference_orientation_rad=orientation,
        calibration_kind=part[KIND_FIELD], status=part[STATUS_FIELD], source_note=part[NOTE_FIELD])


__all__ = ('FIELDS', 'MATRIX_FIELD', 'BIAS_FIELD', 'REFERENCE_ENERGY_FIELD',
    'REFERENCE_ORIENTATION_FIELD', 'STATUS_FIELD', 'NOTE_FIELD', 'KIND_FIELD',
    'CALIBRATION_KIND', 'electrical_channel_count', 'validate_multipole_current_inputs')

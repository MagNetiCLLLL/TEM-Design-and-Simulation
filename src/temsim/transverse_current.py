"""Explicit signed electrical calibration for ideal transverse coil sources.

The saved matrix maps existing optical commands to physical channel currents.
It is never refitted at query time or after a change of beam energy/geometry.
Deflector commands are mrad; stigmator commands are percentages (100 means
100%). Rows and columns follow station order, with X then Y at each station.
"""
from collections.abc import Mapping
from numbers import Integral,Real
import math

PREFIX = 'ideal_transverse_'
MATRIX_FIELD = PREFIX+'current_matrix_a_per_unit'
BIAS_FIELD = PREFIX+'current_bias_a'
REFERENCE_ENERGY_FIELD = PREFIX+'calibration_reference_energy_ev'
KIND_FIELD = PREFIX+'calibration_kind'
STATUS_FIELD = PREFIX+'calibration_status'
NOTE_FIELD = PREFIX+'calibration_source_note'
FIELDS = frozenset((MATRIX_FIELD, BIAS_FIELD, REFERENCE_ENERGY_FIELD, KIND_FIELD, STATUS_FIELD, NOTE_FIELD))
CALIBRATION_KIND = 'native_peak_field_at_reference_planes_v1'


def electrical_channel_count(part):
    from temsim.transverse_winding import transverse_winding_envelopes
    coils = transverse_winding_envelopes(part)
    if not coils:
        raise ValueError('Electrical current inputs require a material transverse winding host')
    return 2*(1+max(coil['station_index'] for coil in coils))


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must contain finite real numbers')
    return float(value)


def validate_transverse_current_inputs(part, by_key=None):
    """Return detached matrix/bias lists, or None for an unconfigured legacy host."""
    if not isinstance(part, Mapping):
        raise ValueError('Transverse electrical inputs require a component mapping')
    supplied={str(key) for key in part if str(key).startswith(PREFIX)}
    if supplied-FIELDS:
        raise ValueError(f"{part.get('key')}: unknown transverse electrical fields: {sorted(supplied-FIELDS)}")
    if not supplied:
        return None
    if part.get('model_3d',{}).get('base',{}).get('kind','existing')!='existing':
        return None # Retain inactive inputs; restoring native CAD validates them.
    required=FIELDS-{BIAS_FIELD}
    if not required.issubset(part):
        raise ValueError('A transverse current calibration requires matrix, reference energy, kind, status and source note')
    from temsim.transverse_winding import TURN_FIELD
    turns=part.get(TURN_FIELD)
    if isinstance(turns,bool) or not isinstance(turns,Integral) or turns<=0:
        raise ValueError('Transverse physical currents require a known positive per-coil turn count')
    count=electrical_channel_count(part)
    matrix=part[MATRIX_FIELD]
    if not isinstance(matrix,(list,tuple)) or len(matrix)!=count or any(
            not isinstance(row,(list,tuple)) or len(row)!=count for row in matrix):
        raise ValueError(f'Transverse current matrix must have shape {count} x {count}')
    matrix=[[_number(value,MATRIX_FIELD) for value in row] for row in matrix]
    bias=part.get(BIAS_FIELD,[0.]*count)
    if not isinstance(bias,(list,tuple)) or len(bias)!=count:
        raise ValueError(f'Transverse current bias must contain {count} values')
    bias=[_number(value,BIAS_FIELD) for value in bias]
    if _number(part[REFERENCE_ENERGY_FIELD],REFERENCE_ENERGY_FIELD)<=0:
        raise ValueError('Transverse calibration reference energy must be positive')
    if part[KIND_FIELD]!=CALIBRATION_KIND:
        raise ValueError('Unsupported transverse current calibration kind')
    if any(not isinstance(part[name],str) or not part[name].strip() for name in (STATUS_FIELD,NOTE_FIELD)):
        raise ValueError('Transverse current calibration requires meaningful status and source note')
    return dict(matrix_a_per_unit=matrix,current_bias_a=bias,
                reference_energy_ev=float(part[REFERENCE_ENERGY_FIELD]),
                calibration_kind=part[KIND_FIELD],status=part[STATUS_FIELD],source_note=part[NOTE_FIELD])


__all__=('FIELDS','MATRIX_FIELD','BIAS_FIELD','REFERENCE_ENERGY_FIELD','STATUS_FIELD','NOTE_FIELD',
         'KIND_FIELD','CALIBRATION_KIND','electrical_channel_count','validate_transverse_current_inputs')

"""Closed internal current groups in an unchanged ideal HP winding pack.

Radial and nested-loop fractions partition complete closed circuits. Nested
fractions interpolate both rectangular loop dimensions; they are not arbitrary
axial pieces of an open conductor. Original normal/skew drives remain separate
from these electrical current ratios. Resistance and power are not modelled.
"""
from collections.abc import Mapping
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
import math
from numbers import Integral, Real
from pathlib import Path
import sys
from threading import RLock

FIELD = 'ideal_transverse_current_segments'
SCHEMA = 'ideal-transverse-current-segments-v1'
SUPPORTED_STATIONS = frozenset(('probe_hp1_hexapole','probe_hp2_hexapole'))
_KEYS = frozenset(('turn_count','radial_fraction','nested_fraction','current_ratio'))
_LAYER_CACHE = OrderedDict()
_LOCK = RLock()


@lru_cache(maxsize=1)
def implementation_identity():
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _real(value,name):
    if isinstance(value,bool) or not isinstance(value,Real):
        raise ValueError(f'{name} must be a finite real number')
    try:result=float(value)
    except (OverflowError,ValueError,TypeError) as error:
        raise ValueError(f'{name} must be a finite real number') from error
    if not math.isfinite(result):raise ValueError(f'{name} must be a finite real number')
    return result


def _bounds(value,name):
    if not isinstance(value,(list,tuple)) or len(value)!=2:
        raise ValueError(f'{name} needs two normalized bounds')
    lo,hi=(_real(v,name) for v in value)
    if not 0<=lo<hi<=1:
        raise ValueError(f'{name} must stay inside the original allocation [0, 1]')
    return (lo,hi)


@dataclass(frozen=True,slots=True)
class TransverseCurrentSegment:
    turn_count: int
    radial_fraction: tuple
    nested_fraction: tuple
    current_ratio: float

    def to_dict(self):
        return dict(turn_count=self.turn_count,radial_fraction=list(self.radial_fraction),
                    nested_fraction=list(self.nested_fraction),current_ratio=self.current_ratio)


def normalize_recipe(recipe):
    if (not isinstance(recipe,Mapping) or set(recipe)!={'schema','segments'}
            or recipe['schema']!=SCHEMA or not isinstance(recipe['segments'],(list,tuple))
            or not 1<=len(recipe['segments'])<=64):
        raise ValueError('Invalid ideal transverse closed-current segment recipe')
    result=[]
    for row in recipe['segments']:
        if not isinstance(row,Mapping) or set(row)!=_KEYS:
            raise ValueError('A transverse current group requires turns, radial/nested bounds and ratio')
        count=row['turn_count']
        if isinstance(count,bool) or not isinstance(count,Integral) or count<=0:
            raise ValueError('A closed current group requires positive integer physical turns')
        cell=TransverseCurrentSegment(int(count),_bounds(row['radial_fraction'],'Radial fraction'),
            _bounds(row['nested_fraction'],'Nested-loop fraction'),_real(row['current_ratio'],'Current ratio'))
        result.append(cell.to_dict())
    return dict(schema=SCHEMA,segments=result)


def bind_recipe(recipe,physical_turns):
    canonical=normalize_recipe(recipe)
    if isinstance(physical_turns,bool) or not isinstance(physical_turns,Integral) or physical_turns<=0:
        raise ValueError('An internal current recipe needs the original positive physical turn count')
    cells=tuple(TransverseCurrentSegment(row['turn_count'],tuple(row['radial_fraction']),
        tuple(row['nested_fraction']),row['current_ratio']) for row in canonical['segments'])
    if sum(c.turn_count for c in cells)!=physical_turns:
        raise ValueError('Internal transverse current groups must preserve the original physical turns')
    radial=sorted({0.,1.,*(v for c in cells for v in c.radial_fraction)})
    for lo,hi in zip(radial[:-1],radial[1:]):
        intervals=sorted(c.nested_fraction for c in cells if c.radial_fraction[0]<=lo and c.radial_fraction[1]>=hi)
        previous=0.
        for a,b in intervals:
            if a!=previous:
                raise ValueError('Closed current regions must tile the full original pack without gaps or overlaps')
            previous=b
        if previous!=1.:
            raise ValueError('Closed current regions must cover the full original pack')
    for cell in cells:
        area=(cell.radial_fraction[1]-cell.radial_fraction[0])*(cell.nested_fraction[1]-cell.nested_fraction[0])
        if not math.isclose(area,cell.turn_count/physical_turns,rel_tol=0.,abs_tol=16*sys.float_info.epsilon):
            raise ValueError('Segment area must preserve the original uniform physical turn density')
    return cells


def radial_recipe(physical_turns,groups=2,*,current_ratios=None):
    if (isinstance(groups,bool) or not isinstance(groups,Integral) or groups<1
            or isinstance(physical_turns,bool) or not isinstance(physical_turns,Integral)
            or physical_turns<groups):
        raise ValueError('Each closed current group needs at least one original physical turn')
    counts=[physical_turns//groups+int(i<physical_turns%groups) for i in range(groups)]
    ratios=[1.]*groups if current_ratios is None else list(current_ratios)
    if len(ratios)!=groups:raise ValueError('Supply one ratio per closed current group')
    previous=0;rows=[]
    for count,ratio in zip(counts,ratios):
        rows.append(dict(turn_count=count,radial_fraction=[previous/physical_turns,(previous+count)/physical_turns],
                         nested_fraction=[0.,1.],current_ratio=ratio))
        previous+=count
    result=normalize_recipe(dict(schema=SCHEMA,segments=rows));bind_recipe(result,physical_turns)
    return result


def set_transverse_current_segments(state,station_key,recipe):
    """Persist an ideal-only electrical edit; geometry and drives stay intact."""
    from temsim.ideal_coil_segments import require_ideal_mode
    require_ideal_mode(state)
    if station_key not in SUPPORTED_STATIONS:
        raise ValueError('Internal transverse current groups currently support installed HP1/HP2 only')
    assembly=getattr(state,'_resolved_assembly',None)
    if assembly is None:raise ValueError('A resolved physical HP assembly is required')
    try:part=assembly.part(station_key)
    except KeyError as error:raise ValueError('Select an installed HP winding') from error
    count=part.data.get('hexapole_winding_turn_count_per_coil')
    canonical=None if recipe is None else normalize_recipe(recipe)
    if canonical is not None:
        cells=bind_recipe(canonical,count)
        from temsim.physics.multipole_coil_field import _bindings, _native_commands
        bindings=_bindings(state,require_current=True,component_keys={station_key})
        if len(bindings)!=1:raise ValueError('An installed calibrated ideal HP current source is required')
        _,component,definition,_,electrical=bindings[0]
        command=_native_commands(component,definition,electrical)
        currents=[sum(float(a)*float(b) for a,b in zip(row,command))+float(bias)
            for row,bias in zip(electrical['matrix_a_per_unit'],electrical['current_bias_a'])]
        for cell in cells:
            if not all(math.isfinite(value*cell.current_ratio*cell.turn_count) for value in currents):
                raise ValueError('Transverse segment ampere-turns exceed the finite numerical range')
    descriptors=deepcopy(state.lens_field_map_descriptors)
    row=dict(descriptors.get(station_key,{}))
    if canonical is None:row.pop(FIELD,None)
    else:row[FIELD]=canonical
    if row:descriptors[station_key]=row
    else:descriptors.pop(station_key,None)
    state.lens_field_map_descriptors=descriptors
    for name in ('_runtime_copper_coil_provider_cache','_runtime_lens_field_provider_cache','_unified_field_runtime_cache'):
        cache=getattr(state,name,None)
        if isinstance(cache,dict):cache.clear()
    return canonical


def apply_to_potential(state,potential):
    from temsim.simulation_modes import mode_key
    if mode_key(state)!='unified':return potential
    if getattr(state,'round_lens_field_model','gaussian')!='copper_coil':return potential
    if potential.source_id not in SUPPORTED_STATIONS:return potential
    recipe=getattr(state,'lens_field_map_descriptors',{}).get(potential.source_id,{}).get(FIELD)
    return potential if recipe is None else replace(potential,current_segments=recipe)


def closed_layers(potential):
    """Complete normal/skew-preserving sublayers; no user drive is a layer."""
    cells=bind_recipe(potential.current_segments,potential.turn_count_per_coil)
    if all(c.current_ratio==1. for c in cells):return (potential,)
    with _LOCK:
        cached=_LAYER_CACHE.get(id(potential))
        if cached is not None and cached[0] is potential:
            _LAYER_CACHE.move_to_end(id(potential));return cached[1]
    layers=[]
    for index,cell in enumerate(cells):
        coils=[]
        for original in potential.coils:
            coil=dict(original)
            lo,hi=cell.radial_fraction
            start=original['radial_start_mm'];length=original['radial_end_mm']-start
            coil['radial_start_mm']=start+lo*length;coil['radial_end_mm']=start+hi*length
            lo,hi=cell.nested_fraction
            for axis in ('width','height'):
                start=original['window_'+axis+'_mm'];length=original['outer_'+axis+'_mm']-start
                coil['window_'+axis+'_mm']=start+lo*length;coil['outer_'+axis+'_mm']=start+hi*length
            coils.append(coil)
        currents=tuple(value*cell.current_ratio for value in potential.currents_a)
        if not all(math.isfinite(v*cell.turn_count) for v in currents):
            raise ValueError('Transverse segment ampere-turns exceed the finite numerical range')
        layers.append(replace(potential,coils=tuple(coils),turn_count_per_coil=cell.turn_count,
            currents_a=currents,current_segments=None))
    result=tuple(layers)
    with _LOCK:
        _LAYER_CACHE[id(potential)]=(potential,result);_LAYER_CACHE.move_to_end(id(potential))
        while len(_LAYER_CACHE)>16:_LAYER_CACHE.popitem(last=False)
    return result


_IMPLEMENTATIONS={name:globals()[name] for name in
    ('_real','_bounds','normalize_recipe','bind_recipe','closed_layers')}


def implementation_is_supported():
    return all(globals().get(name) is function for name,function in _IMPLEMENTATIONS.items())

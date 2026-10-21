"""Explicit material relationships for edits of saved round-lens dimensions.

The caller supplies a private candidate document. Optical centres, excitation,
field calibration and propagation rules are never inferred from these edits.
No missing geometry is synthesized: an incomplete native relationship fails
with an explanation, and an active replacement keeps its old recipe inactive.
"""
import math


FIELDS = frozenset({'pole_gap_mm', 'mechanical_coil_radial_thickness_mm',
                    'mechanical_inner_diameter_mm', 'mechanical_outer_diameter_mm'})
REFERENCE_GAP_KEYS = frozenset({'condenser_lens_1', 'condenser_lens_2'})


def _native(part):
    return part.get('model_3d', {}).get('base', {}).get('kind', 'existing') == 'existing'


def _identity(part):
    return part.get('geometry_template_key', part['key'])


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive dimension')
    return float(value)


def _native_child(part, field):
    if not _native(part) or 'magnetic_radial_profile_mm' in part:
        raise ValueError(f"{part['key']}: {field} cannot locate a native material interface in a replacement or explicit radial profile; edit that material directly")


def _set_extent(part, start, end):
    if not start < end:
        raise ValueError(f"{part['key']}: pole gap leaves no positive pole length")
    part.update(local_start_z_mm=start, local_end_z_mm=end,
                local_center_z_mm=(start + end)/2, length_mm=end-start)


def _coil_for(document, lens):
    coils = [p for p in document['parts'] if p.get('parent_key') == lens['key']
             and p.get('mechanical_profile') == 'magnetic_excitation_coil']
    if len(coils) != 1:
        raise ValueError(f"{lens['key']}: editing radial winding dimensions requires one declared excitation coil")
    return coils[0]


def _synchronize_coil_radials(document, lens, inner, outer):
    if not 0 < inner < outer:
        raise ValueError(f"{lens['key']}: winding requires 0 < material inner diameter < outer diameter")
    owners = [lens]
    if lens.get('shared_housing_key'):
        owners = [p for p in document['parts']
                  if p.get('shared_housing_key') == lens['shared_housing_key']
                  and p.get('mechanical_profile') == 'magnetic_lens_assembly']
    from temsim.cad_round_lens_end_hardware import FAMILY_PREFIXES
    for owner in owners:
        coil = _coil_for(document, owner)
        _native_child(coil, 'winding radial dimensions')
        coil.update(mechanical_inner_diameter_mm=inner, mechanical_outer_diameter_mm=outer)
        # A direct material edit explicitly overrides the legacy design-field
        # sizing fallback with its measured difference, never a guessed value.
        owner['mechanical_coil_radial_thickness_mm'] = (outer-inner)/2
        for housing in document['parts']:
            if housing.get('parent_key') != owner['key'] or not _native(housing):
                continue
            for prefix in FAMILY_PREFIXES:
                if housing.get(prefix+'coil_key') == coil['key']:
                    housing[prefix+'inner_diameter_mm'] = inner
                    housing[prefix+'outer_diameter_mm'] = outer


def synchronize_dimension_edit(document, key, field, old_value):
    """Update saved child material dimensions in the caller's candidate only.

    Two-pole gaps keep the lens centre and the poles' outer/back faces fixed;
    only the facing ends and resulting lengths/centres change. Winding radial
    thickness keeps its material inner diameter fixed and grows/shrinks OD,
    including explicitly coil-matched passive end plates. Existing native
    constraints run before the editor may publish this candidate.
    """
    if field not in FIELDS:
        return
    by_key = {part['key']: part for part in document['parts']}
    row = by_key[key]
    profile = row.get('mechanical_profile')
    direct_coil = profile == 'magnetic_excitation_coil' and field in {
        'mechanical_inner_diameter_mm', 'mechanical_outer_diameter_mm'}
    parent_dimension = profile == 'magnetic_lens_assembly' and field in {
        'pole_gap_mm', 'mechanical_coil_radial_thickness_mm'}
    if not (direct_coil or parent_dimension) or not _native(row):
        return
    value = _positive(row[field], f'{key}.{field}')
    previous = _positive(old_value, f'{key}.{field} previous value')
    if value == previous:
        return
    if direct_coil:
        lens = by_key.get(row.get('parent_key'))
        if lens is None or lens.get('mechanical_profile') != 'magnetic_lens_assembly':
            return  # Independent copied coil has no parent recipe to synchronize.
        _native_child(row, field)
        _synchronize_coil_radials(document, lens, float(row['mechanical_inner_diameter_mm']),
                                 float(row['mechanical_outer_diameter_mm']))
        return
    lens = row
    children = [part for part in document['parts'] if part.get('parent_key') == key]
    if field == 'pole_gap_mm':
        if _identity(lens) in REFERENCE_GAP_KEYS:
            raise ValueError(f'{key}: pole_gap_mm is a reference-only optical dimension; this shared C1/C2 cartridge declares only one pole per channel, so an independent material gap cannot be edited')
        poles = sorted((p for p in children if p.get('mechanical_profile') == 'magnetic_pole_piece'),
                       key=lambda p: float(p['local_start_z_mm']))
        if len(poles) != 2:
            raise ValueError(f'{key}: editing a material pole gap requires two declared pole children')
        upper, lower = poles
        for pole in poles:
            _native_child(pole, field)
        center = float(lens['local_center_z_mm'])
        if (not math.isclose(float(upper['local_end_z_mm']), center-previous/2, abs_tol=1e-8, rel_tol=0)
                or not math.isclose(float(lower['local_start_z_mm']), center+previous/2, abs_tol=1e-8, rel_tol=0)):
            raise ValueError(f'{key}: saved pole faces do not bound the previous gap about its fixed centre; edit the declared pole geometry explicitly')
        _set_extent(upper, float(upper['local_start_z_mm']), center-value/2)
        _set_extent(lower, center+value/2, float(lower['local_end_z_mm']))
        # The production section builder also checks explicit noses/shanks.
        from temsim.part_model_3d import _pole_section
        for pole in poles:
            _pole_section(pole, pole['local_start_z_mm'], pole['local_end_z_mm'])
    else:
        coil = _coil_for(document, lens)
        _native_child(coil, field)
        inner, outer = (float(coil[name]) for name in ('mechanical_inner_diameter_mm', 'mechanical_outer_diameter_mm'))
        if not math.isclose((outer-inner)/2, previous, abs_tol=1e-8, rel_tol=0):
            raise ValueError(f'{key}: saved coil diameters do not match the previous radial thickness; edit the declared coil geometry explicitly')
        _synchronize_coil_radials(document, lens, inner, inner+2*value)
    # The document transaction next checks each native construction. Keep
    # buildable inter-component conflicts as drafts for the overlap list;
    # full assembly validation still gates Save, as before this binding.

"""Material editors distinguish physical dimensions from retained references."""
from copy import deepcopy

import pytest

from temsim.parameter_impact import describe_parameter_impact
from temsim.part_model_3d import part_dimension_specs, part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


@pytest.mark.parametrize('module,key,field', [
    ('column/C3_ProbeCorrector.toml', 'condenser_lens_1', 'pole_gap_mm'),
    ('column/C3_ProbeCorrector.toml', 'condenser_lens_2', 'pole_gap_mm'),
    *[('column/C3_ProbeCorrector.toml', key, 'mechanical_bore_diameter_mm') for key in (
        'condenser_aperture_2', 'condenser_aperture_3', 'objective_aperture')],
    ('project_and_recording_system/NoEnergyFilter.toml', 'selected_area_aperture', 'mechanical_bore_diameter_mm'),
    ('project_and_recording_system/EnergyFilter.toml', 'energy_filter_entrance_aperture', 'mechanical_bore_diameter_mm'),
    ('project_and_recording_system/NoEnergyFilter.toml', 'diffraction_lens_upper_pole', 'pole_cone_angle_to_axis_deg'),
])
def test_nonmaterial_reference_cannot_pretend_to_resize_a_surface(module, key, field):
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT / module)
    before = deepcopy(draft.document)
    spec = next(s for s in part_dimension_specs(draft.document, key) if s.path == ('parts', key, field))
    assert not spec.editable and 'reference' in spec.reason.lower()
    impact = describe_parameter_impact(draft.part(key), spec.path)
    assert impact.status == 'inactive' and 'geometry' not in impact.effects
    with pytest.raises(ValueError, match='[Rr]eference'):
        draft.set_dimension(spec.path, spec.value * 1.01)
    assert draft.document == before and not draft.dirty and not draft.can_undo


def test_replacement_exposes_its_active_dimensions_and_retains_inactive_recipe():
    draft = PartModelDocument(INSTRUMENT_CONFIG_ROOT / 'column/C3_ProbeCorrector.toml')
    key = draft.copy_component_from(draft.document, 'beam_deflector', 'detached', 2500., include_children=False)
    native = deepcopy(draft.part(key))
    draft.set_model_3d(key, dict(schema_version=1, base=dict(kind='box', width_mm=8., height_mm=6., length_mm=1.)))
    specs = {s.path: s for s in part_dimension_specs(draft.document, key)}
    path = ('parts', key, 'model_3d', 'base', 'width_mm')
    assert specs[path].editable
    old = part_model_from_document(draft.document, key).meshes[0].vertices.max(0)
    draft.set_dimension(path, 10.)
    new = part_model_from_document(draft.document, key).meshes[0].vertices.max(0)
    assert tuple(old) != tuple(new)
    inactive = ('parts', key, 'deflector_support_radial_skin_thickness_mm')
    assert not specs[inactive].editable and 'inactive' in specs[inactive].reason
    assert draft.part(key)['mechanical_outer_diameter_mm'] == native['mechanical_outer_diameter_mm']

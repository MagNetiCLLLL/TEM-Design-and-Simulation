"""Source-policy checks only; no coherent or finite-source calculation."""
from types import SimpleNamespace

import pytest

from temsim.module_manifest import read_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.physics.transverse_coil_field import TransverseCoilPotential
from temsim.physics.transverse_coil_wave_policy import require_transverse_coil_wave_admission
from temsim.physics.wave_field_admission import (
    require_supported_column_wave_fields, require_supported_wave_dipoles,
)
from temsim.transverse_winding import transverse_winding_envelopes


def test_unqualified_actual_current_potential_rejects_before_field_query(monkeypatch):
    parts=read_document(INSTRUMENT_CONFIG_ROOT/'column/C3_ProbeCorrector.toml')['parts']
    part=next(p for p in parts if p['key']=='condenser_stigmator')
    potential=TransverseCoilPotential('condenser_stigmator',
        transverse_winding_envelopes(part),400,(0.,0.))
    def forbidden(*args,**kwargs):
        pytest.fail('Admission must precede field evaluation or wave work')
    monkeypatch.setattr(TransverseCoilPotential,'sample',forbidden)
    snapshot=SimpleNamespace(contributions=(SimpleNamespace(potential=potential),))
    with pytest.raises(ValueError,match='classical particles only'):
        require_transverse_coil_wave_admission(snapshot)


def test_native_providers_keep_their_existing_separate_admission():
    require_transverse_coil_wave_admission(SimpleNamespace(
        contributions=(SimpleNamespace(potential=object()),)))


@pytest.mark.parametrize('wrapped',[False,True])
def test_dynamic_current_rows_are_rejected_without_freezing_or_substituting(wrapped):
    from test_transverse_coil_field import dynamic
    potential=dynamic()
    row=SimpleNamespace(potential=potential) if wrapped else potential
    with pytest.raises(ValueError,match='native multipoles cannot be substituted'):
        require_transverse_coil_wave_admission(SimpleNamespace(contributions=(row,)))


@pytest.mark.parametrize('guard',[require_supported_column_wave_fields,require_supported_wave_dipoles])
def test_old_wave_plan_routes_reject_ideal_mode_before_plan_or_provider_work(guard):
    state=SimpleNamespace(simulation_mode='unified',round_lens_field_model='copper_coil')
    with pytest.raises(ValueError,match='General/downstream coherent propagation is not qualified'):
        guard(state,0.,1.,None)

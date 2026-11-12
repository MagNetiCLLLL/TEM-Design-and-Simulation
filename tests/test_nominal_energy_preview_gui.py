"""Nominal preview restores the user's source sampling draft without editing physics."""
from test_coherent_result_labels import observation, preview_profile
from temsim.gui.coherent_result_labels import coherent_preview_text






def test_saved_nominal_preview_is_labeled_as_unresolved_energy_distribution():
    profile = {**preview_profile(), "source_nominal_energy_preview": True}
    label = coherent_preview_text(observation(profile))
    assert "Nominal emission energy (preview)" in label
    assert "configured FWHM not integrated" in label
    assert "not accuracy qualified" in label

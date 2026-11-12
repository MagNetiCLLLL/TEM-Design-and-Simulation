"""Offscreen explicit phase-preparation control and honest estimate labels."""
from dataclasses import replace

from test_coherent_result_labels import observation, preview_profile
from temsim.gui.coherent_result_labels import coherent_preview_text






def test_result_label_separates_estimated_representation_error_from_projection_bound():
    original = preview_profile()
    assert "representation" not in coherent_preview_text(observation(original))
    profile = {**original, "column_preview_phase_reselection": True,
        "phase_preparation_maximum_pixels": 4096,
        "estimated_representation_l2_per_reference": .0002,
        "combined_projection_and_estimated_l2_per_reference": .0012,
        "representation_estimates_certified": False}
    label = coherent_preview_text(observation(profile))
    for text in ("not accuracy qualified", "Column cap 2048", "CPU phase preparation cap 4096",
                 "projection L2 cost 0.001", "representation L2 estimate 0.0002",
                 "combined L2 cost 0.0012", "estimated, not a certified bound"):
        assert text in label

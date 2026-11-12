"""Offscreen control transitions; no physical propagation is substituted as evidence."""
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from temsim.immutable_json import json_digest
from temsim.physics.tip_wave_pipeline import TipWaveRequest












@pytest.mark.parametrize("value", [1, 0, "true", None, np.bool_(True)])
def test_request_rejects_nonboolean_weighted_resampling(value):
    with pytest.raises(ValueError, match="explicit boolean"):
        replace(TipWaveRequest(), column_support_projection=True, column_weighted_resampling=value).validate()


def test_request_requires_controlled_support_for_weighted_resampling():
    with pytest.raises(ValueError, match="requires controlled support"):
        replace(TipWaveRequest(), column_weighted_resampling=True).validate()
    replace(TipWaveRequest(), column_support_projection=True, column_weighted_resampling=True).validate()






def test_exploratory_preview_result_label_contains_actual_losses_without_accuracy_claim():
    from types import SimpleNamespace
    from temsim.gui.coherent_beam import _coherent_preview_text
    checkpoint = SimpleNamespace(record={"upstream": {"unified_numerical_profile": {
        "exploratory_preview": True, "effective_column_maximum_pixels": 1024,
        "maximum_support_probability_fraction": .001, "maximum_accumulated_support_l2": .1,
        "numerical_projection_loss_per_reference": 2.5e-5,
        "complex_l2_projection_bound_per_reference": .015}}})
    text = _coherent_preview_text(checkpoint)
    for phrase in ("Exploratory preview", "not accuracy qualified", "1024 pixels/axis",
                   "probability 0.001", "L2 0.1", "projection loss 2.5e-05", "projection L2 cost 0.015"):
        assert phrase in text
    assert _coherent_preview_text(SimpleNamespace(record={})) == ""

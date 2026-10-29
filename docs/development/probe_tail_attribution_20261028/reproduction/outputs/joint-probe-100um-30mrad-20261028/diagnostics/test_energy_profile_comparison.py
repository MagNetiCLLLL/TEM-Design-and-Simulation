"""Pure-array checks for saved energy-profile comparisons."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parent))
from compare_column_energy_diagnostic import paired


def fixture():
    a=dict(source_id=np.array([306,307]),weights=np.array([.25,.75]),z_mm=np.array([1.,2.,3.]),
        x_m=np.array([[0.,1.],[2.,3.],[4.,5.]])*1e-9,y_m=np.zeros((3,2)),alive=np.ones((3,2),bool))
    return (Path('.'),{},a,'hash')


def test_translation_is_not_removed_from_raw_paired_comparison():
    a=fixture();b=deepcopy(a);b[2]['x_m']+=3e-9;b[2]['y_m']+=4e-9
    report=paired(a,b,2.)
    assert report['surface']['max_position_difference_nm']==pytest.approx(5.)
    assert report['exact_shared_planes']==3


def test_population_interception_is_explicit_not_silently_dropped():
    a=fixture();b=deepcopy(a);b[2]['alive'][1:,0]=False
    report=paired(a,b,2.)
    assert report['surface']['paired_live_points']==1
    assert report['surface']['changed_survival_points']==1


def test_changed_absolute_weight_rejects_pairing():
    a=fixture();b=deepcopy(a);b[2]['weights']*=2
    with pytest.raises(ValueError,match='weights'):paired(a,b,2.)


def test_nonfinite_live_coordinate_rejects_pairing():
    a=fixture();b=deepcopy(a);b[2]['x_m'][1,0]=np.nan
    with pytest.raises(ValueError,match='finite'):paired(a,b,2.)

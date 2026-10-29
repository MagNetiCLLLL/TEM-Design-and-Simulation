"""Low-cost regression checks; no particle propagation."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parent))
from analyze_launch_momentum import linear_fit


def test_heldout_targets_do_not_change_fitted_coefficients():
    rng=np.random.default_rng(84);x=rng.normal(size=(60,4));y=x@rng.normal(size=(4,4))+.2
    train=np.arange(60)%5!=1;w=np.full(60,1/60)
    pred,fit=linear_fit(x,y,w,train)
    altered=y.copy();altered[~train]+=1000
    pred2,fit2=linear_fit(x,altered,w,train)
    np.testing.assert_array_equal(pred,pred2)
    assert fit==fit2
    np.testing.assert_allclose(pred[~train],y[~train],rtol=1e-13,atol=1e-13)


def test_heldout_features_do_not_change_training_scaling():
    rng=np.random.default_rng(85);x=rng.normal(size=(60,4));y=x@rng.normal(size=(4,4))
    train=np.arange(60)%5!=1;w=np.full(60,1/60)
    _,fit=linear_fit(x,y,w,train)
    x2=x.copy();x2[~train]*=1e9
    _,fit2=linear_fit(x2,y,w,train)
    assert fit==fit2


def test_rank_deficient_fit_rejected():
    x=np.arange(20.)[:,None]*np.ones((1,4));y=x.copy()
    with pytest.raises(ValueError,match='Rank-deficient'):
        linear_fit(x,y,np.ones(20),np.ones(20,bool))

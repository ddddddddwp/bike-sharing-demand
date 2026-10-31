import numpy as np

from src.common.metrics import rmsle, mae, rmse


def test_rmsle_known():
    y = np.array([0.0, 1.0, 10.0])
    p = np.array([0.0, 1.0, 10.0])
    assert rmsle(y, p) == 0.0
    assert rmsle(y, p + 1.0) > 0.0


def test_metrics_nonnegative_clip():
    y = np.array([1.0, 2.0])
    p = np.array([-5.0, 2.0])
    assert mae(y, p) == 0.5
    assert rmse(y, p) == np.sqrt(0.5)

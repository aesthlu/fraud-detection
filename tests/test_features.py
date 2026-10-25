import numpy as np
import pandas as pd

from fraud_detection.features import FeatureBuilder


def test_drops_raw_time_and_amount(transactions):
    out = FeatureBuilder().fit_transform(transactions.drop(columns="Class"))

    assert "Time" not in out.columns
    assert "Amount" not in out.columns
    assert {"hour", "Amount_log"} <= set(out.columns)


def test_hour_wraps_around_each_day():
    X = pd.DataFrame({"Time": [0, 3600 * 5, 3600 * 23, 3600 * 24, 3600 * 29], "Amount": [1.0] * 5})

    out = FeatureBuilder().transform(X)

    assert out["hour"].tolist() == [0, 5, 23, 0, 5]


def test_amount_log_handles_zero():
    X = pd.DataFrame({"Time": [0, 0], "Amount": [0.0, np.e - 1]})

    out = FeatureBuilder().transform(X)

    assert np.allclose(out["Amount_log"], [0.0, 1.0])


def test_does_not_modify_input(transactions):
    X = transactions.drop(columns="Class")
    before = X.copy()

    FeatureBuilder().transform(X)

    pd.testing.assert_frame_equal(X, before)

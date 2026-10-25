import pytest

from fraud_detection.ingestion import load_data, temporal_split


def test_load_data_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(str(tmp_path / "absent.csv"))


def test_temporal_split_sizes(transactions):
    train, val, test = temporal_split(transactions, train_ratio=0.6, val_ratio=0.2)

    assert len(train) + len(val) + len(test) == len(transactions)
    assert len(train) == pytest.approx(0.6 * len(transactions), rel=0.01)
    assert len(val) == pytest.approx(0.2 * len(transactions), rel=0.02)


def test_temporal_split_never_mixes_past_and_future(transactions):
    """Le cœur du split temporel : aucune transaction de validation/test n'est antérieure au train."""
    train, val, test = temporal_split(transactions, train_ratio=0.6, val_ratio=0.2)

    assert train["Time"].max() < val["Time"].min()
    assert val["Time"].max() < test["Time"].min()

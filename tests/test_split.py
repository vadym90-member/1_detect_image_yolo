import pytest

from src.dataset.split import SplitRatios, assign_splits, group_by_split


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        SplitRatios(0.5, 0.2, 0.2)


def test_ratios_reject_negative():
    with pytest.raises(ValueError, match="Negative ratio"):
        SplitRatios(1.1, -0.05, -0.05)


def test_assignment_is_deterministic():
    keys = [f"img_{i:04d}" for i in range(500)]
    ratios = SplitRatios(0.8, 0.1, 0.1)

    a = assign_splits(keys, ratios)
    b = assign_splits(keys, ratios)
    assert a == b


def test_assignment_distribution_is_reasonable():
    keys = [f"img_{i:04d}" for i in range(2000)]
    ratios = SplitRatios(0.8, 0.1, 0.1)

    grouped = group_by_split(assign_splits(keys, ratios))
    n = len(keys)

    assert abs(len(grouped["train"]) / n - 0.8) < 0.03
    assert abs(len(grouped["val"]) / n - 0.1) < 0.03
    assert abs(len(grouped["test"]) / n - 0.1) < 0.03
    assert len(grouped["train"]) + len(grouped["val"]) + len(grouped["test"]) == n


def test_changing_salt_changes_assignments():
    keys = [f"img_{i:04d}" for i in range(200)]
    ratios = SplitRatios(0.8, 0.1, 0.1)

    a = assign_splits(keys, ratios, salt="A")
    b = assign_splits(keys, ratios, salt="B")
    assert a != b

import pytest

from engine.decision_engine import DecisionEngine

select = DecisionEngine._select_banded_target


@pytest.mark.parametrize("direction", [1, -1])
def test_t1_selected_in_band(direction):
    price = 24000.0
    target, label = select(price, [price + direction * 50, price + direction * 100], direction)
    assert label == "T1"
    assert target == price + direction * 50


@pytest.mark.parametrize("direction", [1, -1])
def test_t1_band_edges(direction):
    price = 24000.0
    assert select(price, [price + direction * 47], direction)[1] == "T1"
    assert select(price, [price + direction * 53], direction)[1] == "T1"


@pytest.mark.parametrize("direction", [1, -1])
def test_t2_selected_when_no_t1(direction):
    price = 24000.0
    target, label = select(price, [price + direction * 30, price + direction * 105], direction)
    assert label == "T2"
    assert target == price + direction * 105


@pytest.mark.parametrize("direction", [1, -1])
def test_fallback_when_outside_bands(direction):
    price = 24000.0
    candidates = [None, price + direction * 40, price + direction * 70, price - direction * 50]
    target, label = select(price, candidates, direction)
    assert label == "T1_FALLBACK"
    assert target == price + direction * 50


def test_fallback_with_no_candidates():
    assert select(100.0, [], 1) == (150.0, "T1_FALLBACK")

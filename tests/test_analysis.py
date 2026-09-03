import numpy as np

from apoptotic_control.analysis import first_contiguous_block, first_zero_crossing
from apoptotic_control.parameters import (
    FIGURE_2_THRESHOLD,
    FIGURE_3,
    FIGURE_3_MAX_ITER,
    FIGURE_S4,
    FIGURE_S7,
    FIGURE_S8_CONSTRAINED,
    FIGURE_S8_QUADRATIC,
)


def test_parameters_match_the_figure_runs():
    assert FIGURE_2_THRESHOLD["n_actions"] == 201
    assert FIGURE_3 == {
        "N": 10000,
        "r": 0.30,
        "delta": 0.15,
        "delta0": 0.05,
        "alpha": 0.51,
        "penalty": 1e4,
        "n_actions": 201,
    }
    assert FIGURE_3_MAX_ITER == 3500
    assert FIGURE_S4["N"] == 800
    assert FIGURE_S8_QUADRATIC["N"] == 1500
    assert FIGURE_S8_CONSTRAINED["initial_state"] == 1


def test_s7_uses_its_own_model_sizes_and_budgets():
    assert FIGURE_S7["threshold"]["N"] == 500
    assert FIGURE_S7["quadratic"]["N"] == 500
    assert FIGURE_S7["unbounded"]["N"] == 800
    assert FIGURE_S7["constrained"]["N"] == 400
    assert FIGURE_S7["constrained"]["beta"] == 0.35
    assert FIGURE_S7["constrained_quadratic"]["N"] == 400
    assert FIGURE_S7["constrained_quadratic"]["beta"] == 0.37


def test_zero_crossing_uses_the_original_midpoint_rule():
    states = np.arange(5)
    drift = np.array([np.nan, 0.4, 0.1, -0.3, -0.5])
    assert first_zero_crossing(states, drift) == 2.5


def test_first_contiguous_occupied_block_stops_at_the_first_gap():
    mask = np.array([False, True, True, False, True, True])
    expected = np.array([False, True, True, False, False, False])
    np.testing.assert_array_equal(first_contiguous_block(mask), expected)

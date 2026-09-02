import numpy as np

from apoptotic_control.models import CTMDPTumorBase
from apoptotic_control.solvers import (
    solve_constrained,
    solve_constrained_quadratic,
    solve_quadratic,
    solve_threshold,
    solve_unbounded,
)


def common():
    return {
        "N": 20,
        "r": 0.30,
        "delta": 0.15,
        "delta0": 0.05,
        "alpha": 0.51,
        "penalty": 1e4,
        "n_actions": 11,
    }


def test_rates_and_zero_drift_action():
    model = CTMDPTumorBase()
    model.load_common_params(common())
    assert np.isclose(model.a_star, 0.20 / 0.45)
    assert np.isclose(model.lam(10, 0.5), 1.5)
    assert np.isclose(model.mu(10, 0.5), 1.25)
    assert np.isclose(model.drift(10, model.a_star), 0.0)


def test_policy_iteration_models():
    threshold = dict(common(), L=10, kappa1=1.0, kappa2=0.5)
    quadratic = dict(common(), c1=0.01, c2=0.0, kappa=1.3, c3=0.75)
    unbounded = dict(common(), kappa=1.0, c3=1.0)
    for model, _, policy in (
        solve_threshold(threshold),
        solve_quadratic(quadratic),
        solve_unbounded(unbounded),
    ):
        assert len(policy) == model.N + 1
        assert np.all((policy >= 0) & (policy <= 1))


def test_constrained_models():
    linear = dict(common(), kappa=2.0, c3=1.0, initial_state=10)
    quadratic = dict(common(), kappa=1.3, c1=0.01, c2=0.0, c3=0.75, initial_state=10)
    for result in (
        solve_constrained(linear, beta=1.0),
        solve_constrained_quadratic(quadratic, beta=1.0),
    ):
        assert result["result"].success
        assert np.isfinite(result["occupancy"]).all()
        assert result["support"].any()

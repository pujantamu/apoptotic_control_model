COMMON = {
    "r": 0.30,
    "delta": 0.15,
    "delta0": 0.05,
    "penalty": 1e4,
}


FIGURE_1 = {
    "shared": {
        "N": 100,
        "r": 0.20,
        "delta": 0.25,
        "delta0": 0.05,
        "alpha": 0.51,
        "penalty": 1e4,
    },
    "threshold": {"L": 35, "kappa1": 1.0, "kappa2": 0.5, "n_actions": 101},
    "quadratic": {
        "c1": 0.005,
        "c2": 0.02,
        "kappa": 1.2,
        "c3": 1.0,
        "n_actions": 101,
    },
    "unbounded": {"N": 800, "kappa": 1.0, "c3": 0.8, "n_actions": 201},
    "constrained": {
        "N": 200,
        "kappa": 1.2,
        "c3": 1.0,
        "beta": 0.35,
        "n_actions": 101,
        "initial_state": 10,
    },
    "constrained_quadratic": {
        "N": 200,
        "kappa": 1.3,
        "c1": 0.01,
        "c2": 0.0,
        "c3": 0.75,
        "beta": 0.37,
        "n_actions": 101,
        "initial_state": 10,
    },
}


FIGURE_2_THRESHOLD = {
    **COMMON,
    "N": 200,
    "alpha": 0.50,
    "L": 50,
    "kappa1": 1.0,
    "kappa2": 5.0,
    "n_actions": 101,
}

FIGURE_2_QUADRATIC = {
    **COMMON,
    "N": 250,
    "alpha": 0.05,
    "c1": 0.005,
    "c2": 0.02,
    "kappa": 2.0,
    "c3": 1.0,
    "n_actions": 201,
}

FIGURE_3 = {
    "N": 10000,
    "r": 0.30,
    "delta": 0.20,
    "delta0": 0.0,
    "alpha": 0.51,
    "penalty": 1e4,
    "n_actions": 201,
}

FIGURE_3_REGIMES = (
    (50.0, 0.5, r"strong regularization: $\kappa=50,\ c_3=.5$"),
    (1.0, 1.0, r"balanced: $\kappa=1,\ c_3=1$"),
    (1.0, 5.0, r"growth-dominated: $\kappa=1,\ c_3=5$"),
)

FIGURE_4_CONSTRAINED = {
    **COMMON,
    "N": 200,
    "alpha": 0.51,
    "kappa": 2.0,
    "c3": 1.0,
    "n_actions": 101,
    "initial_state": 10,
}

FIGURE_4_CONSTRAINED_QUADRATIC = {
    **COMMON,
    "N": 200,
    "alpha": 0.51,
    "kappa": 1.3,
    "c1": 0.01,
    "c2": 0.0,
    "c3": 0.75,
    "n_actions": 101,
    "initial_state": 10,
}

FIGURE_4_BETAS = (0.01, 0.38, 1.1, 2.0, 3.0, 4.0, 5.0)
FIGURE_4_QUADRATIC_BETAS = (0.05, 0.30, 0.50, 0.75, 1.0, 1.2, 1.5)


FIGURE_S1 = {
    **COMMON,
    "N": 200,
    "alpha": 0.50,
    "L": 50,
    "kappa1": 1.0,
    "kappa2": 1.0,
    "n_actions": 101,
}

FIGURE_S2 = FIGURE_2_QUADRATIC.copy()

FIGURE_S4 = {
    **COMMON,
    "N": 500,
    "alpha": 0.51,
    "kappa": 1.0,
    "c3": 1.0,
    "n_actions": 201,
}

FIGURE_S5 = FIGURE_4_CONSTRAINED.copy()
FIGURE_S6 = FIGURE_4_CONSTRAINED_QUADRATIC.copy()

FIGURE_S8_QUADRATIC = {
    **COMMON,
    "N": 1500,
    "alpha": 0.51,
    "kappa": 1.3,
    "c1": 0.01,
    "c2": 0.0,
    "c3": 0.75,
    "n_actions": 101,
    "initial_state": 10,
}

FIGURE_S8_CONSTRAINED = FIGURE_S8_QUADRATIC.copy()
FIGURE_S8_BETA = 1.2

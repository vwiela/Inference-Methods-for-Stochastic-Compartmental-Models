import numpy as np
from bayesflow.simulation import Prior
from .utils import unconstrain, constrain, unconstrain_lognormal, constrain_lognormal

RNG = np.random.default_rng(2025)


def seir2v_prior_full():
    """Generates a random draw from the joint prior."""

    beta = RNG.uniform(low=0, high=1)
    gamma_inv = RNG.normal(15.7, 6.7)
    kappa_inv = RNG.lognormal(mean=1.63, sigma=0.5)
    t = RNG.uniform(low=120, high=360)
    s = RNG.uniform(low=0.1, high=10)
    I0 = RNG.uniform(low=10, high=1000)

    while True:
        if gamma_inv > 1:
            break
        gamma_inv = RNG.normal(15.7, 6.7)

    while True:
        if gamma_inv * beta >= 0.95 and gamma_inv * beta <= 4:
            break
        beta = RNG.uniform(low=0, high=1)

    return np.array([gamma_inv, kappa_inv, beta, s, t, I0])


def seir2v_prior_reparam():
    """Generates a random draw from the joint prior."""

    r0 = RNG.uniform(low=0.95, high=4)
    e0 = RNG.uniform(low=6, high=30)
    s0 = RNG.uniform(low=1, high=100)
    t = RNG.uniform(low=120, high=360)
    I0 = RNG.uniform(low=10, high=1000)

    return np.array([r0, e0, s0, t, I0])


def seir2v_prior_full_uniform():
    """Generates a random draw from the joint prior."""

    beta = RNG.uniform(low=0, high=1)
    gamma_inv = RNG.uniform(low=1, high=30)
    kappa_inv = RNG.uniform(low=1, high=30)
    s = RNG.uniform(low=0.1, high=10)
    t = RNG.uniform(low=120, high=360)
    I0 = RNG.uniform(low=10, high=1000)

    while True:
        if gamma_inv * beta >= 0.95 and gamma_inv * beta <= 4:
            break
        beta = RNG.uniform(low=0, high=1)

    return np.array([gamma_inv, kappa_inv, beta, s, t, I0])


def seir2v_prior_reparam_nonuniform():
    """Generates a random draw from the joint prior."""

    r0 = RNG.normal(2.2, 0.4)
    e0 = RNG.normal(20, 5)
    s0 = RNG.uniform(low=1, high=100)
    t = RNG.uniform(low=120, high=360)
    I0 = RNG.uniform(low=10, high=1000)

    while True:
        if r0 > 0.95:
            break
        r0 = RNG.normal(2.2, 0.4)

    while True:
        if e0 > 5:
            break
        e0 = RNG.normal(20, 5)

    return np.array([r0, e0, s0, t, I0])


def sir_prior():
    """Generates a random draw from the joint prior."""

    beta = RNG.uniform(low=0, high=1)
    gamma = RNG.uniform(low=1, high=30.0)

    while True:
        if gamma * beta >= 0.95 and gamma * beta <= 5:
            break
        beta = RNG.uniform(low=0, high=1)
    return np.array([beta, gamma])


def sir_prior2():
    """Generates a random draw from the joint prior."""

    beta = RNG.normal(0.15, 0.03)
    gamma = RNG.normal(15, 3)

    while True:
        if beta > 0:
            break
        beta = RNG.normal(0.15, 0.03)

    while True:
        if gamma > 0:
            break
        gamma = RNG.normal(15, 3)
    return np.array([beta, gamma])


def normal():
    """Generates a random draw from the joint prior."""

    beta = RNG.normal(0, 1)
    gamma = RNG.normal(0, 1)

    return np.array([beta, gamma])


def seir2v_prior_full_transform(params):
    params[..., 1] = unconstrain_lognormal(params[..., 1])
    params[..., 2] = unconstrain(params[..., 2], 0, 1)
    params[..., 3] = unconstrain(params[..., 3], 0.1, 10)
    params[..., 4] = unconstrain(params[..., 4], 120, 360)
    params[..., 5] = unconstrain(params[..., 5], 10, 1000)
    return params


def seir2v_prior_full_test_transform(params):
    params[..., 1] = unconstrain(params[..., 1], 0, 1)
    params[..., 2] = unconstrain(params[..., 2], 0.1, 10)
    params[..., 3] = unconstrain(params[..., 3], 120, 360)
    params[..., 4] = unconstrain(params[..., 4], 10, 1000)
    return params


def seir2v_prior_reparam_transform(params):
    params[..., 0] = unconstrain(params[..., 0], 0.95, 4)
    params[..., 1] = unconstrain(params[..., 1], 6, 30)
    params[..., 2] = unconstrain(params[..., 2], 1, 100)
    params[..., 3] = unconstrain(params[..., 3], 120, 360)
    params[..., 4] = unconstrain(params[..., 4], 10, 1000)
    return params


def seir2v_prior_full_uniform_transform(params):
    params[..., 0] = unconstrain(params[..., 0], 1, 30)
    params[..., 1] = unconstrain(params[..., 1], 1, 30)
    params[..., 2] = unconstrain(params[..., 2], 0, 1)
    params[..., 3] = unconstrain(params[..., 3], 0.1, 10)
    params[..., 4] = unconstrain(params[..., 4], 120, 360)
    params[..., 5] = unconstrain(params[..., 5], 10, 1000)
    return params


def seir2v_prior_reparam_nonuniform_transform(params):
    params[..., 2] = unconstrain(params[..., 2], 1, 100)
    params[..., 3] = unconstrain(params[..., 3], 120, 360)
    params[..., 4] = unconstrain(params[..., 4], 10, 1000)
    return params


def sir_prior_transform(params):
    params[..., 0] = unconstrain(params[..., 0], 0, 1)
    params[..., 1] = unconstrain(params[..., 1], 1, 30)
    return params


def sir_prior2_transform(params):
    return params


def seir2v_prior_full_transform_inv(params):
    params[..., 1] = constrain_lognormal(params[..., 1])
    params[..., 2] = constrain(params[..., 2], 0, 1)
    params[..., 3] = constrain(params[..., 3], 0.1, 10)
    params[..., 4] = constrain(params[..., 4], 120, 360)
    params[..., 5] = constrain(params[..., 5], 10, 1000)
    return params


def seir2v_prior_full_test_transform_inv(params):
    params[..., 1] = constrain(params[..., 1], 0, 1)
    params[..., 2] = constrain(params[..., 2], 0.1, 10)
    params[..., 3] = constrain(params[..., 3], 120, 360)
    params[..., 4] = constrain(params[..., 4], 10, 1000)
    return params


def seir2v_prior_reparam_transform_inv(params):
    params[..., 0] = constrain(params[..., 0], 0.95, 4)
    params[..., 1] = constrain(params[..., 1], 6, 30)
    params[..., 2] = constrain(params[..., 2], 1, 100)
    params[..., 3] = constrain(params[..., 3], 120, 360)
    params[..., 4] = constrain(params[..., 4], 10, 1000)
    return params


def seir2v_prior_full_uniform_transform_inv(params):
    params[..., 0] = constrain(params[..., 0], 1, 30)
    params[..., 1] = constrain(params[..., 1], 1, 30)
    params[..., 2] = constrain(params[..., 2], 0, 1)
    params[..., 3] = constrain(params[..., 3], 0.1, 10)
    params[..., 4] = constrain(params[..., 4], 120, 360)
    params[..., 5] = constrain(params[..., 5], 10, 1000)
    return params


def seir2v_prior_reparam_nonuniform_transform_inv(params):
    params[..., 2] = constrain(params[..., 2], 1, 100)
    params[..., 3] = constrain(params[..., 3], 120, 360)
    params[..., 4] = constrain(params[..., 4], 10, 1000)
    return params


def sir_prior_transform_inv(params):
    params[..., 0] = constrain(params[..., 0], 0, 1)
    params[..., 1] = constrain(params[..., 1], 1, 30)
    return params


def sir_prior2_transform_inv(params):
    return params


def seir2v_prior_full_wrap():
    prior = Prior(
        prior_fun=seir2v_prior_full,
        param_names=["$\\gamma^{-1}$", "$\\kappa^{-1}$", "$\\beta$", "$s$", "$t_{var}$", "$I_0$"],
    )

    params = prior.__call__(1000)["prior_draws"]
    params = seir2v_prior_full_transform(params)
    prior_means = np.mean(params, axis=0, keepdims=True)
    prior_stds = np.std(params, axis=0, keepdims=True)

    prior_dict = {
        "prior": prior,
        "prior_means": prior_means,
        "prior_stds": prior_stds,
        "transform": seir2v_prior_full_transform,
        "transform_inv": seir2v_prior_full_transform_inv,
    }
    return prior_dict


def seir2v_prior_reparam_wrap():
    prior = Prior(
        prior_fun=seir2v_prior_reparam,
        param_names=["$r_{0}$", "$e_{0}$", "$s_{0}$", "$t_{var}$", "$I_0$"],
    )

    """Calculate prior_means and prior_stds to normalize prior draws"""
    params = prior.__call__(1000)["prior_draws"]
    params = seir2v_prior_reparam_transform(params)
    prior_means = np.mean(params, axis=0, keepdims=True)
    prior_stds = np.std(params, axis=0, keepdims=True)

    prior_dict = {
        "prior": prior,
        "prior_means": prior_means,
        "prior_stds": prior_stds,
        "transform": seir2v_prior_reparam_transform,
        "transform_inv": seir2v_prior_reparam_transform_inv,
    }
    return prior_dict


def seir2v_prior_full_uniform_wrap():
    prior = Prior(
        prior_fun=seir2v_prior_full_uniform,
        param_names=["$\\gamma^{-1}$", "$\\kappa^{-1}$", "$\\beta$", "$s$", "$t_{var}$", "$I_0$"],
    )

    """Calculate prior_means and prior_stds to normalize prior draws"""
    params = prior.__call__(1000)["prior_draws"]
    params = seir2v_prior_full_uniform_transform(params)
    prior_means = np.mean(params, axis=0, keepdims=True)
    prior_stds = np.std(params, axis=0, keepdims=True)

    prior_dict = {
        "prior": prior,
        "prior_means": prior_means,
        "prior_stds": prior_stds,
        "transform": seir2v_prior_full_uniform_transform,
        "transform_inv": seir2v_prior_full_uniform_transform_inv,
    }
    return prior_dict


def seir2v_prior_reparam_nonuniform_wrap():
    prior = Prior(
        prior_fun=seir2v_prior_reparam_nonuniform,
        param_names=["$r_{0}$", "$e_{0}$", "$s_{0}$", "$t_{var}$", "$I_0$"],
    )

    """Calculate prior_means and prior_stds to normalize prior draws"""

    params = prior.__call__(1000)["prior_draws"]
    params = seir2v_prior_reparam_nonuniform_transform(params)
    prior_means = np.mean(params, axis=0, keepdims=True)
    prior_stds = np.std(params, axis=0, keepdims=True)
    prior_dict = {
        "prior": prior,
        "prior_means": prior_means,
        "prior_stds": prior_stds,
        "transform": seir2v_prior_reparam_nonuniform_transform,
        "transform_inv": seir2v_prior_reparam_nonuniform_transform_inv,
    }
    return prior_dict


def sir_prior_wrap():
    prior = Prior(prior_fun=sir_prior, param_names=["$\\beta$", "$\\gamma^{-1}$"])

    """Calculate prior_means and prior_stds to normalize prior draws"""
    # prior_means, prior_stds = prior.estimate_means_and_stds()

    params = prior.__call__(1000)["prior_draws"]
    params = sir_prior_transform(params)
    prior_means = np.mean(params, axis=0, keepdims=True)
    prior_stds = np.std(params, axis=0, keepdims=True)

    prior_dict = {
        "prior": prior,
        "prior_means": prior_means,
        "prior_stds": prior_stds,
        "transform": sir_prior_transform,
        "transform_inv": sir_prior_transform_inv,
    }
    return prior_dict


def sir_prior2_wrap():
    prior = Prior(prior_fun=sir_prior2, param_names=["$\\beta$", "$\\gamma^{-1}$"])

    """Calculate prior_means and prior_stds to normalize prior draws"""
    # prior_means, prior_stds = prior.estimate_means_and_stds()

    params = prior.__call__(1000)["prior_draws"]
    params = sir_prior2_transform(params)
    prior_means = np.mean(params, axis=0, keepdims=True)
    prior_stds = np.std(params, axis=0, keepdims=True)

    prior_dict = {
        "prior": prior,
        "prior_means": prior_means,
        "prior_stds": prior_stds,
        "transform": sir_prior2_transform,
        "transform_inv": sir_prior2_transform_inv,
    }
    return prior_dict

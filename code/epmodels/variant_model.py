import numpy as np
from scipy.stats import binom, truncnorm
import math

from julia import Main
import os

import random
from random import sample

_cached_variant_model = None
_variant_model_initialized = False


def _load_variant_model(N, T):
    global _variant_model_initialized, _cached_variant_model
    if not _variant_model_initialized:
        jl_path = os.path.join(os.path.dirname(__file__), "virus_variant_est_infc_2.jl")
        Main.include(jl_path)
        _variant_model_initialized = True
        _cached_variant_model = Main.SEIR_Variant_SDE(N, T)
    return _cached_variant_model


def seir2v_forward(params, N0=180000, T1=400):
    """Performs a forward simulation from the SEIR model with mutation given a random draw from the prior."""
    var_infc = 100
    step = 10
    params.append(var_infc)

    model = _load_variant_model(N0, T1)
    model_output = model(params)
    E_wt, S, R_var, E_both, I_wt, I_both, E_var, R_both, R_wt, I_var = (
        np.zeros((T1 + 1, 1)),
        np.zeros((T1 + 1, 1)),
        np.zeros((T1 + 1, 1)),
        np.zeros((T1 + 1, 1)),
        np.zeros((T1 + 1, 1)),
        np.zeros((T1 + 1, 1)),
        np.zeros((T1 + 1, 1)),
        np.zeros((T1 + 1, 1)),
        np.zeros((T1 + 1, 1)),
        np.zeros((T1 + 1, 1)),
    )
    for i in range(T1 + 1):
        E_wt[i] = model_output[1][step * i][0]
        S[i] = model_output[1][step * i][1]
        R_var[i] = model_output[1][step * i][2]
        E_both[i] = model_output[1][step * i][3]
        I_wt[i] = model_output[1][step * i][4]
        I_both[i] = model_output[1][step * i][5]
        E_var[i] = model_output[1][step * i][6]
        R_both[i] = model_output[1][step * i][7]
        R_wt[i] = model_output[1][step * i][8]
        I_var[i] = model_output[1][step * i][9]

    # print("E_wt", E_wt)
    # print("I_wt", model_output[1][:][4])

    arr = np.array([E_wt, S, R_var, E_both, I_wt, I_both, E_var, R_both, R_wt, I_var])
    N_t = arr.sum(axis=0)

    return E_wt, S, R_var, E_both, I_wt, I_both, E_var, R_both, R_wt, I_var, N_t


def seir2v_gaussian_dense(params, timepoints, std1, std2, T1=400):
    """Calls model_forward to get compartments to calculate observations via a additive gaussian observational model"""

    n_draws, n_params = params.shape

    out = np.zeros((n_draws, len(timepoints), 3))

    for n in range(n_draws):

        if n_params == 6:
            gamma_inv, kappa_inv, beta, s, t1, I0 = params[n]
        if n_params == 5:
            r0, e0, s0, t1, I0 = params[n]
            kappa_inv = 5
            gamma_inv = e0 - kappa_inv
            beta = r0 / gamma_inv
            s = s0 / gamma_inv

        t1 = int(t1)

        E_wt, S, R_var, E_both, I_wt, I_both, E_var, R_both, R_wt, I_var, N_t = seir2v_forward(
            [gamma_inv, kappa_inv, beta, t1, I0]
        )
        timepoints_scaled = np.zeros((len(timepoints), 1))
        data1 = np.zeros((len(timepoints), 1))
        data2 = np.zeros((len(timepoints), 1))

        for i in range(len(timepoints)):
            timepoints_scaled[i] = timepoints[i] / T1
            t = timepoints[i]

            loc1 = s * (I_wt[t] + I_var[t] + I_both[t]) / N_t[t]
            scale1 = std1[i]
            data1[i] = truncnorm.rvs(-loc1 / scale1, (1 - loc1) / scale1, loc=loc1, scale=scale1)

            loc2 = (R_wt[t] + R_var[t] + R_both[t] + E_both[t] + I_both[t]) / N_t[t]
            scale2 = std2[i]
            data2[i] = truncnorm.rvs(-loc2 / scale2, (1 - loc2) / scale2, loc=loc2, scale=scale2)

        out[n] = np.concatenate([timepoints_scaled, data1, data2], axis=-1)

    return out


def seir2v_gaussian_sparse(params, timepoints, std1, std2, T1=400, config=None):
    """Calls model_forward to get compartments to calculate observations via a additive gaussian observational model"""

    n_draws, n_params = params.shape
    out = np.zeros((n_draws, len(timepoints), 5))

    for n in range(n_draws):
        if n_params == 6:
            gamma_inv, kappa_inv, beta, s, t1, I0 = params[n]
        if n_params == 5:
            r0, e0, s0, t1, I0 = params[n]
            kappa_inv = 5
            gamma_inv = e0 - kappa_inv
            beta = r0 / gamma_inv
            s = s0 / gamma_inv

        t1 = int(t1)

        E_wt, S, R_var, E_both, I_wt, I_both, E_var, R_both, R_wt, I_var, N_t = seir2v_forward(
            [gamma_inv, kappa_inv, beta, t1, I0]
        )

        timepoints_scaled = np.zeros((len(timepoints), 1))
        data1 = np.zeros((len(timepoints), 1))
        missing1 = np.zeros((len(timepoints), 1))
        data2 = np.zeros((len(timepoints), 1))
        missing2 = np.zeros((len(timepoints), 1))

        # number_timepoints1 = random.randint(4, len(timepoints))
        timepoints1 = config[
            "timepoints1_nonmissing"
        ]  # sample(list(timepoints), number_timepoints1)

        # number_timepoints2 = random.randint(4, len(timepoints))
        timepoints2 = config[
            "timepoints2_nonmissing"
        ]  # sample(list(timepoints), number_timepoints2)

        for i in range(len(timepoints)):
            timepoints_scaled[i] = timepoints[i] / T1
            t = timepoints[i]

            if t in timepoints1:
                loc1 = s * (I_wt[t] + I_var[t] + I_both[t]) / N_t[t]
                scale1 = std1[i]
                data1[i] = truncnorm.rvs(
                    -loc1 / scale1, (1 - loc1) / scale1, loc=loc1, scale=scale1
                )
                missing1[i] = 1
            else:
                data1[i] = -1
                missing1[i] = 0

            if t in timepoints2:
                loc2 = (R_wt[t] + R_var[t] + R_both[t] + E_both[t] + I_both[t]) / N_t[t]
                scale2 = std2[i]
                data2[i] = truncnorm.rvs(
                    -loc2 / scale2, (1 - loc2) / scale2, loc=loc2, scale=scale2
                )
                missing2[i] = 1
            else:
                data2[i] = -1
                missing2[i] = 0

        out[n] = np.concatenate([timepoints_scaled, data1, missing1, data2, missing2], axis=-1)
    return out


def seir2v_binomial_dense(params, timepoints, samplesize1, samplesize2, T1=400):
    """Calls model_forward to get compartments to calculate observations via a binomial observational model"""

    n_draws, n_params = params.shape

    out = np.zeros((n_draws, len(timepoints), 3))

    for n in range(n_draws):
        if n_params == 6:
            gamma_inv, kappa_inv, beta, s, t1, I0 = params[n]
        if n_params == 5:
            r0, e0, s0, t1, I0 = params[n]
            kappa_inv = 5
            gamma_inv = e0 - kappa_inv
            beta = r0 / gamma_inv
            s = s0 / gamma_inv

        t1 = int(t1)

        E_wt, S, R_var, E_both, I_wt, I_both, E_var, R_both, R_wt, I_var, N_t = seir2v_forward(
            [gamma_inv, kappa_inv, beta, t1, I0]
        )

        timepoints_scaled = np.zeros((len(timepoints), 1))
        data1 = np.zeros((len(timepoints), 1))
        data2 = np.zeros((len(timepoints), 1))

        for i in range(len(timepoints)):
            timepoints_scaled[i] = timepoints[i] / T1
            t = timepoints[i]

            p1 = np.clip((s * (I_wt[t] + I_var[t] + I_both[t])) / N_t[t], 0.0, 1.0).item()
            n1 = samplesize1[i]
            data1[i] = binom.rvs(n1, p1) / n1

            p2 = np.clip(
                (R_wt[t] + R_var[t] + R_both[t] + E_both[t] + I_both[t]) / N_t[t], 0.0, 1.0
            ).item()
            n2 = samplesize2[i]
            data2[i] = binom.rvs(n2, p2) / n2

        out[n] = np.concatenate([timepoints_scaled, data1, data2], axis=-1)

    return out


def seir2v_binomial_sparse(params, timepoints, samplesize1, samplesize2, T1=400):
    """Calls model_forward to get compartments to calculate observations via a binomial observational model"""

    n_draws, n_params = params.shape

    out = np.zeros((n_draws, len(timepoints), 5))

    for n in range(n_draws):
        if n_params == 6:
            gamma_inv, kappa_inv, beta, s, t1, I0 = params[n]
        if n_params == 5:
            r0, e0, s0, t1, I0 = params[n]
            kappa_inv = 5
            gamma_inv = e0 - kappa_inv
            beta = r0 / gamma_inv
            s = s0 / gamma_inv

        t1 = int(t1)

        E_wt, S, R_var, E_both, I_wt, I_both, E_var, R_both, R_wt, I_var, N_t = seir2v_forward(
            [gamma_inv, kappa_inv, beta, t1, I0]
        )

        timepoints_scaled = np.zeros((len(timepoints), 1))
        data1 = np.zeros((len(timepoints), 1))
        missing1 = np.zeros((len(timepoints), 1))
        data2 = np.zeros((len(timepoints), 1))
        missing2 = np.zeros((len(timepoints), 1))

        number_timepoints1 = random.randint(4, len(timepoints))
        timepoints1 = sample(list(timepoints), number_timepoints1)

        number_timepoints2 = random.randint(4, len(timepoints))
        timepoints2 = sample(list(timepoints), number_timepoints2)

        for i in range(len(timepoints)):
            timepoints_scaled[i] = timepoints[i] / T1
            t = timepoints[i]

            if t in timepoints1:
                p1 = np.clip((s * (I_wt[t] + I_var[t] + I_both[t])) / N_t[t], 0.0, 1.0).item()
                n1 = samplesize1[i]
                data1[i] = binom.rvs(n1, p1) / n1
                missing1[i] = 1
            else:
                data1[i] = -1
                missing1[i] = 0

            if t in timepoints2:
                p2 = np.clip(
                    (R_wt[t] + R_var[t] + R_both[t] + E_both[t] + I_both[t]) / N_t[t], 0.0, 1.0
                ).item()
                n2 = samplesize2[i]
                data2[i] = binom.rvs(n2, p2) / n2
                missing2[i] = 1
            else:
                data2[i] = -1
                missing2[i] = 0

        out[n] = np.concatenate([timepoints_scaled, data1, missing1, data2, missing2], axis=-1)

    return out


def seir2v_binomial(params, batch_size, config):
    """Calls model_forward to get compartments to calculate observations via a binomial observational model"""
    """Helper function for plots"""

    n_draws, n_params = params.shape

    out1 = np.zeros((n_draws, batch_size, len(config["timepoints1_nonmissing"])))
    out2 = np.zeros((n_draws, batch_size, len(config["timepoints2_nonmissing"])))

    assert n_params == 6 or n_params == 5, "n_params of `params` needs to be 5 or 6"

    for n in range(n_draws):
        if n_params == 6:
            gamma_inv, kappa_inv, beta, s, t1, I0 = params[n]
        if n_params == 5:
            r0, e0, s0, t1, I0 = params[n]
            kappa_inv = 5
            gamma_inv = e0 - kappa_inv
            beta = r0 / gamma_inv
            s = s0 / gamma_inv

        t1 = int(t1)

        for batch in range(batch_size):
            E_wt, S, R_var, E_both, I_wt, I_both, E_var, R_both, R_wt, I_var, N_t = seir2v_forward(
                [gamma_inv, kappa_inv, beta, t1, I0]
            )

            data1 = np.zeros((len(config["timepoints1_nonmissing"])))
            data2 = np.zeros((len(config["timepoints2_nonmissing"])))

            for i in range(len(config["timepoints1_nonmissing"])):
                t = config["timepoints1_nonmissing"][i]
                p1 = np.clip((s * (I_wt[t] + I_var[t] + I_both[t])) / N_t[t], 0.0, 1.0).item()
                n1 = config["samplesize1"][i]
                data1[i] = binom.rvs(n1, p1) / n1

            for i in range(len(config["timepoints2_nonmissing"])):
                t = config["timepoints2_nonmissing"][i]
                p2 = np.clip(
                    (R_wt[t] + R_var[t] + R_both[t] + E_both[t] + I_both[t]) / N_t[t], 0.0, 1.0
                ).item()
                n2 = config["samplesize2"][i]
                data2[i] = binom.rvs(n2, p2) / n2

            out1[n, batch] = data1
            out2[n, batch] = data2

    return out1, out2


def seir2v_gaussian(params, batch_size, config):
    """Calls model_forward to get compartments to calculate observations via a binomial observational model"""
    """Helper function for plots"""

    n_draws, n_params = params.shape

    out1 = np.zeros((n_draws, batch_size, len(config["timepoints1_nonmissing"])))
    out2 = np.zeros((n_draws, batch_size, len(config["timepoints2_nonmissing"])))

    assert n_params == 6 or n_params == 5, "n_params of `params` needs to be 5 or 6"

    for n in range(n_draws):
        if n_params == 6:
            gamma_inv, kappa_inv, beta, s, t1, I0 = params[n]
        if n_params == 5:
            r0, e0, s0, t1, I0 = params[n]
            kappa_inv = 5
            gamma_inv = e0 - kappa_inv
            beta = r0 / gamma_inv
            s = s0 / gamma_inv

        t1 = int(t1)

        for batch in range(batch_size):
            E_wt, S, R_var, E_both, I_wt, I_both, E_var, R_both, R_wt, I_var, N_t = seir2v_forward(
                [gamma_inv, kappa_inv, beta, t1, I0]
            )

            data1 = np.zeros((len(config["timepoints1_nonmissing"])))
            data2 = np.zeros((len(config["timepoints2_nonmissing"])))

            for i in range(len(config["timepoints1_nonmissing"])):
                t = config["timepoints1_nonmissing"][i]
                loc1 = s * (I_wt[t] + I_var[t] + I_both[t]) / N_t[t]
                scale1 = config["std1"][i]
                data1[i] = truncnorm.rvs(
                    -loc1 / scale1, (1 - loc1) / scale1, loc=loc1, scale=scale1
                )

            for i in range(len(config["timepoints2_nonmissing"])):
                t = config["timepoints2_nonmissing"][i]
                loc2 = (R_wt[t] + R_var[t] + R_both[t] + E_both[t] + I_both[t]) / N_t[t]
                scale2 = config["std2"][i]
                data2[i] = truncnorm.rvs(
                    -loc2 / scale2, (1 - loc2) / scale2, loc=loc2, scale=scale2
                )

            out1[n, batch] = data1
            out2[n, batch] = data2

    return out1, out2


def seir2v(params, T1=400):
    """Calls model_forward to get compartments to calculate observations without an observation model"""
    """Helper function for plots"""
    batch_size = params.shape[0]
    n_params = params.shape[1]

    out1 = np.zeros((batch_size, T1 + 1, 1))
    out2 = np.zeros((batch_size, T1 + 1, 1))

    for batch in range(batch_size):
        infc = np.zeros((T1 + 1, 1))
        prev = np.zeros((T1 + 1, 1))

        gamma_inv, kappa_inv, beta, s, t1, I0 = params[batch]
        t1 = int(t1)

        E_wt, S, R_var, E_both, I_wt, I_both, E_var, R_both, R_wt, I_var, N_t = seir2v_forward(
            [gamma_inv, kappa_inv, beta, t1, I0]
        )

        for t in range(T1 + 1):
            infc[t] = (s * (I_wt[t] + I_var[t] + I_both[t])) / N_t[t]
            prev[t] = (R_wt[t] + R_var[t] + R_both[t] + E_both[t] + I_both[t]) / N_t[t]

        out1[batch] = infc
        out2[batch] = prev

    return out1, out2


def generative_model_seir2v(batch_size, config, prior_dict, observationtype="binomial"):
    """Get batch_size prior draws"""
    params = prior_dict["prior"].__call__(batch_size)["prior_draws"]

    if config["type"] == "dense":
        if observationtype == "binomial":
            sim_data = seir2v_binomial_dense(
                params,
                timepoints=config["timepoints"],
                samplesize1=config["samplesize1"],
                samplesize2=config["samplesize2"],
            )
        if observationtype == "gaussian" or observationtype == "normal":
            sim_data = seir2v_gaussian_dense(
                params, timepoints=config["timepoints"], std1=config["std1"], std2=config["std2"]
            )
    if config["type"] == "sparse":
        if observationtype == "binomial":
            sim_data = seir2v_binomial_sparse(
                params,
                timepoints=config["timepoints"],
                samplesize1=config["samplesize1"],
                samplesize2=config["samplesize2"],
            )
        if observationtype == "gaussian" or observationtype == "normal":
            sim_data = seir2v_gaussian_sparse(
                params,
                timepoints=config["timepoints"],
                std1=config["std1"],
                std2=config["std2"],
                config=config,
            )
    out_dict = {"prior_draws": params, "sim_data": sim_data}
    return out_dict


def data_gen_seir2v(params, timepoints, samplesize1, samplesize2, T1=400):
    """Calls model_forward to get compartments to calculate observations via a binomial observational model"""

    n_draws, n_params = params.shape

    out = np.zeros((n_draws, len(timepoints), 5))

    for n in range(n_draws):
        if n_params == 6:
            gamma_inv, kappa_inv, beta, s, t1, I0 = params[n]
        if n_params == 5:
            r0, e0, s0, t1, I0 = params[n]
            kappa_inv = 5
            gamma_inv = e0 - kappa_inv
            beta = r0 / gamma_inv
            s = s0 / gamma_inv

        t1 = int(t1)

        E_wt, S, R_var, E_both, I_wt, I_both, E_var, R_both, R_wt, I_var, N_t = seir2v_forward(
            [gamma_inv, kappa_inv, beta, t1, I0]
        )

        timepoints_scaled = np.zeros((len(timepoints), 1))
        data1 = np.zeros((len(timepoints), 1))
        data2 = np.zeros((len(timepoints), 1))
        std1 = np.zeros((len(timepoints), 1))
        std2 = np.zeros((len(timepoints), 1))

        for i in range(len(timepoints)):
            timepoints_scaled[i] = timepoints[i] / T1
            t = timepoints[i]

            p1 = np.clip((s * (I_wt[t] + I_var[t] + I_both[t])) / N_t[t], 0.0, 1.0).item()
            n1 = samplesize1[i]
            data1[i] = binom.rvs(n1, p1) / n1
            std1[i] = math.sqrt(p1 * (1 - p1) / n1)

            p2 = np.clip(
                (R_wt[t] + R_var[t] + R_both[t] + E_both[t] + I_both[t]) / N_t[t], 0.0, 1.0
            ).item()
            n2 = samplesize2[i]
            data2[i] = binom.rvs(n2, p2) / n2
            std2[i] = math.sqrt(p2 * (1 - p2) / n2)

        out[n] = np.concatenate([timepoints_scaled, data1, std1, data2, std2], axis=-1)

    return out

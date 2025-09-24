import numpy as np
from scipy.stats import binom, truncnorm
import platform
import math


from julia import Main
import os

from random import sample

_cached_sir_model = None
_sir_model_initialized = False
system = platform.system()

def _load_sir_model(N,T):
    global _sir_model_initialized, _cached_sir_model
    if not _sir_model_initialized:
        if system == "Windows":
            jl_path = os.path.join(os.path.dirname(__file__), "sir_model.jl")
        if system == "Linux":
            jl_path = os.path.join(os.path.dirname(__file__), "sir_model_2.jl")

        Main.include(jl_path)
        _sir_model_initialized = True
        _cached_sir_model = Main.SIR_SDE(N, T)
    return _cached_sir_model

def sir_forward(params, N0=180000, T1=100):
    """Performs a forward simulation from the SIR model with mutation given a random draw from the prior."""
    step = 10
    if _sir_model_initialized is False:
        print("SIR model not initialized.")
    model = _load_sir_model(N0,T1)
    model_output = model(params)
    S, I = np.zeros((T1+1, 1)), np.zeros((T1+1, 1))
    for i in range(T1+1):
        S[i] = model_output[1][step * i][0]
        I[i] = model_output[1][step * i][1]

    return S, I


def sir_gaussian_dense(params, timepoints, std1, std2, T1=100):   
    """Calls model_forward to get compartments to calculate observations via a additive gaussian observational model"""

    n_draws, n_params = params.shape

    out = np.zeros((n_draws, len(timepoints),3))

    for param in range(n_draws):

        beta, gamma = params[param]
        #beta = beta / gamma
        S, I = sir_forward([beta, 1 / gamma])

        timepoints_scaled = np.zeros((len(timepoints),1))
        data1 = np.zeros((len(timepoints),1))
        data2 = np.zeros((len(timepoints),1))

        for i in range(len(timepoints)):
            timepoints_scaled[i] = timepoints[i] / T1
            t = timepoints[i]

            loc1 = np.clip(I[t], 0.0, 1.0).item()
            scale1  = std1[i]
            data1[i] = truncnorm.rvs(-loc1/scale1, (1-loc1)/scale1, loc=loc1, scale=scale1)

            loc2 = np.clip(1 - S[t], 0.0, 1.0).item()
            scale2 = std2[i]
            data2[i] = truncnorm.rvs(-loc2/scale2, (1-loc2)/scale2, loc=loc2, scale=scale2)

        out[param] = np.concatenate([timepoints_scaled, data1, data2], axis = -1)
        
    return out


def sir_binomial_dense(params, timepoints, samplesize1, samplesize2, T1=100):  
    """Calls model_forward to get compartments to calculate observations via a additive gaussian observational model"""

    n_draws, n_params = params.shape

    out = np.zeros((n_draws, len(timepoints),3))

    for param in range(n_draws):

        beta, gamma = params[param]
        #beta = beta / gamma
        S, I = sir_forward([beta, 1 / gamma])

        timepoints_scaled = np.zeros((len(timepoints),1))
        data1 = np.zeros((len(timepoints),1))
        data2 = np.zeros((len(timepoints),1))

        for i in range(len(timepoints)):
            timepoints_scaled[i] = timepoints[i] / T1
            t = timepoints[i]

            p1 = np.clip(I[t], 0.0, 1.0).item()
            n1  = samplesize1[i]
            data1[i] = binom.rvs(n1, p1) / n1

            p2 = np.clip(1 - S[t], 0.0, 1.0).item()
            n2 = samplesize2[i]
            data2[i] = binom.rvs(n2, p2) / n2

        out[param] = np.concatenate([timepoints_scaled, data1, data2], axis = -1)
        
    return out

def sir_gaussian(params, batch_size, config):
    """Calls model_forward to get compartments to calculate observations via a additive gaussian observational model"""

    n_draws, n_params = params.shape

    out1 = np.zeros((n_draws, batch_size, len(config["timepoints1_nonmissing"])))
    out2 = np.zeros((n_draws, batch_size, len(config["timepoints2_nonmissing"])))

    for n in range(n_draws):
        beta, gamma = params[n]
        #beta = beta / gamma

        for batch in range(batch_size):  
            S, I = sir_forward([beta, 1 / gamma])

            data1 = np.zeros((len(config["timepoints1_nonmissing"])))
            data2 = np.zeros((len(config["timepoints2_nonmissing"])))

            for i in range(len(config["timepoints1_nonmissing"])):
                t = config["timepoints1_nonmissing"][i]
                loc1 = np.clip(I[t], 0.0, 1.0).item()
                scale1  = config["std1"][i]
                data1[i] = truncnorm.rvs(-loc1/scale1, (1-loc1)/scale1, loc=loc1, scale=scale1)

            for i in range(len(config["timepoints2_nonmissing"])):
                t = config["timepoints2_nonmissing"][i]           
                loc2 = np.clip(1 - S[t], 0.0, 1.0).item()
                scale2 = config["std2"][i]
                data2[i] = truncnorm.rvs(-loc2/scale2, (1-loc2)/scale2, loc=loc2, scale=scale2)
            
            out1[n,batch] = data1
            out2[n,batch] = data2

    return out1, out2


def sir_binomial(params, batch_size, config):  
    """Calls model_forward to get compartments to calculate observations via a additive gaussian observational model"""

    n_draws, n_params = params.shape

    out1 = np.zeros((n_draws, batch_size, len(config["timepoints1_nonmissing"])))
    out2 = np.zeros((n_draws, batch_size, len(config["timepoints2_nonmissing"])))

    for n in range(n_draws):
        beta, gamma = params[n]
        #beta = beta / gamma

        for batch in range(batch_size):  
            S, I = sir_forward([beta, 1 / gamma])

            data1 = np.zeros((len(config["timepoints1_nonmissing"])))
            data2 = np.zeros((len(config["timepoints2_nonmissing"])))

            for i in range(len(config["timepoints1_nonmissing"])):
                t = config["timepoints1_nonmissing"][i]
                p1 = np.clip(I[t], 0.0, 1.0).item()
                n1  = config["samplesize1"][i]
                data1[i] = binom.rvs(n1, p1) / n1

            for i in range(len(config["timepoints2_nonmissing"])):
                t = config["timepoints2_nonmissing"][i]           
                p2 = np.clip(1 - S[t], 0.0, 1.0).item()
                n2 = config["samplesize2"][i]
                data2[i] = binom.rvs(n2, p2) / n2
            
            out1[n,batch] = data1
            out2[n,batch] = data2

    return out1, out2

def generative_model_sir(batch_size, config, prior_dict, observationtype = "binomial"):
    """Get batch_size prior draws"""
    params = prior_dict["prior"].__call__(batch_size)["prior_draws"]

    if (config["type"] == "dense"):
        if (observationtype == "binomial"): 
            sim_data = sir_binomial_dense(params, timepoints = config["timepoints"], samplesize1 = config["samplesize1"], samplesize2 = config["samplesize2"])
        if (observationtype == "gaussian" or observationtype == "normal"): 
            sim_data = sir_gaussian_dense(params, timepoints = config["timepoints"], std1 = config["std1"], std2 = config["std2"])
    out_dict = {"prior_draws": params, "sim_data": sim_data}
    return out_dict

def data_gen_sir(params, timepoints, samplesize1, samplesize2, T1=100):  
    """Calls model_forward to get compartments to calculate observations via a additive gaussian observational model"""

    n_draws, n_params = params.shape

    out = np.zeros((n_draws, len(timepoints),5))

    for param in range(n_draws):

        beta, gamma = params[param]
        #beta = beta / gamma
        S, I = sir_forward([beta, 1 / gamma])

        timepoints_scaled = np.zeros((len(timepoints),1))
        data1 = np.zeros((len(timepoints),1))
        data2 = np.zeros((len(timepoints),1))
        std1 = np.zeros((len(timepoints),1))
        std2 = np.zeros((len(timepoints),1))    

        for i in range(len(timepoints)):
            timepoints_scaled[i] = timepoints[i] / T1       
            t = timepoints[i]

            p1 = np.clip(I[t], 0.0, 1.0).item()
            n1  = samplesize1[i]
            data1[i] = binom.rvs(n1, p1) / n1
            std1[i] = math.sqrt(p1*(1-p1)/n1)

            p2 = np.clip(1 - S[t], 0.0, 1.0).item()
            n2 = samplesize2[i]
            data2[i] = binom.rvs(n2, p2) / n2
            std2[i] = math.sqrt(p2*(1-p2)/n2)

        out[param] = np.concatenate([timepoints_scaled, data1, std1, data2, std2], axis = -1)
        
    return out
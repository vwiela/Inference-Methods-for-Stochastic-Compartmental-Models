import numpy as np
import pandas as pd
from scipy.stats import nbinom, binom, uniform, truncnorm

from random import sample


def load_data_dense(filepath, dataname, T1 = 400, N0 = 180000):
    """Helper function to load data."""
    data = np.array(pd.read_csv(filepath, sep=","))
    
    timepoints = data[:,1].astype(int)
    obs1 = data[:,2]
    std1 = data[:,3]
    samplesize1 = data[:,4].astype(int)
    missing1 = data[:,5].astype(int)
    obs2 = data[:,6]
    std2 = data[:,7]
    samplesize2 = data[:,8].astype(int)
    missing2 = data[:,9].astype(int)

    """Scale timepoints to use in neural net"""
    timepoints_scaled = timepoints / T1

    """Extract non-missing data and their timepoints"""
    obs1_nonmissing = data[data[:,5] == 1][:,2]
    obs2_nonmissing = data[data[:,9] == 1][:,6] 

    timepoints1_nonmissing = data[data[:,5] == 1][:,1].astype(int)
    timepoints2_nonmissing = data[data[:,9] == 1][:,1].astype(int)

    """For dense data we only use timepoints_scaled, obs_1 and obs_2 as input"""
    data = np.stack([timepoints_scaled, obs1, obs2], axis=-1)
    data = data[np.newaxis,:]

    out_dict = {"T" : T1,
                "N" : N0,
                "type" : "dense",
                "dataname": dataname,
                "timepoints" : timepoints,
                "std1" : std1,
                "samplesize1" : samplesize1,
                "std2" : std2,
                "samplesize2" : samplesize2,
                "obs1_nonmissing" : obs1_nonmissing,
                "obs2_nonmissing" : obs2_nonmissing,
                "timepoints1_nonmissing" : timepoints1_nonmissing,
                "timepoints2_nonmissing" : timepoints2_nonmissing,
                "obs_data" : data
                }

    return out_dict

def load_data_sparse(filepath, dataname, T1 = 400, N0 = 180000):
    """Helper function to load data."""
    data = np.array(pd.read_csv(filepath, sep=","))
    
    timepoints = data[:,1].astype(int)
    obs1 = data[:,2]
    std1 = data[:,3]
    samplesize1 = data[:,4].astype(int)
    missing1 = data[:,5].astype(int)
    obs2 = data[:,6]
    std2 = data[:,7]
    samplesize2 = data[:,8].astype(int)
    missing2 = data[:,9].astype(int)

    """Scale timepoints to use in neural net"""
    timepoints_scaled = timepoints / T1

    """Extract non-missing data and their timepoints"""
    obs1_nonmissing = data[data[:,5] == 1][:,2]
    obs2_nonmissing = data[data[:,9] == 1][:,6] 

    timepoints1_nonmissing = data[data[:,5] == 1][:,1].astype(int)
    timepoints2_nonmissing = data[data[:,9] == 1][:,1].astype(int)

    """For sparse data we use timepoints_scaled, obs_1, missing1, obs_2 and missing2 as input"""
    data = np.stack([timepoints_scaled, obs1, missing1, obs2, missing2], axis=-1)
    data = data[np.newaxis,:]

    out_dict = {"T" : T1,
                "N" : N0,
                "type" : "sparse",
                "dataname" : dataname,
                "timepoints" : timepoints,
                "std1" : std1,
                "samplesize1" : samplesize1,
                "std2" : std2,
                "samplesize2" : samplesize2,
                "obs1_nonmissing" : obs1_nonmissing,
                "obs2_nonmissing" : obs2_nonmissing,
                "timepoints1_nonmissing" : timepoints1_nonmissing,
                "timepoints2_nonmissing" : timepoints2_nonmissing,
                "obs_data" : data
                }

    return out_dict
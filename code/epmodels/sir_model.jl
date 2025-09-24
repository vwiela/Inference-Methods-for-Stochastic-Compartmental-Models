using LinearAlgebra
using Random
using StaticArrays
using Distributions
using DifferentialEquations
using StochasticDiffEq

## Using SBML and SBMLToolkit directly
using SBML 
using SBMLToolkit
using Catalyst
using SymbolicUtils

include("utils/posEM.jl")


function SIR_SDEProblem(parammap, N; endtime=100.0, initial_state=nothing)
    
    # set the timespan
    tspan = (0., endtime)
    
    @parameters β γ
    @variables t s(t) i(t)
    D = Differential(t)
    
    drift = [D(s) ~ -β*s*i,
          D(i) ~ β*s*i-γ*i]
    
    diffusion = [sqrt(β*s*i/N) 0 ;-sqrt(β*s*i/N) sqrt(γ*i/N)]
    
    @named sde = SDESystem(drift, diffusion, t, [s, i], [β, γ])

    if isnothing(parammap)
        parammap = [β=>0.2, γ=>0.05]
    end
    
    # define SDE problem
    if isnothing(initial_state)
        sdeprob = SDEProblem(sde, [], tspan, parammap)
    else
        sdeprob = SDEProblem(sde, initial_state, tspan, parammap)
    end


    return sdeprob

end

function SIR_Model_Simulation(sdeprob, params)
    """
    Function to simulate the SIR model. 
    It returns the timepoints and corresponding state-vectors of the system.
    """
    if params != sdeprob.p[1:2]
        sdeprob = remake(sdeprob, p=params)
    end

    solve_alg = PositiveEM()
    solve_kwargs = (dt=1e-1, dense=true, force_dtmin=true)

    sde_sol = solve(sdeprob, solve_alg; solve_kwargs...)

    return sde_sol.t, sde_sol.u
end

function SIR_SDE(N, T; initial_infc=1800)
    """
    Function to create the SIR model.
    - params: parameter vector of the model in the ordering 
        [beta, gamma]
    - N: total population size
    - T: endtime of the simulation
    - initial_infc: initial number of infected individuals
    """

    endtime = T 
    initial_SuS = N - initial_infc
    initial_state = [initial_SuS/N, initial_infc/N]
    sdeprob = SIR_SDEProblem(nothing, N, endtime=endtime, initial_state=initial_state)

    return params -> SIR_Model_Simulation(sdeprob, params)
end

function SIR_SDE_Ensemble(N, T; initial_infc=1800)
    """
    Function to create the SIR model.
    - params: parameter vector of the model in the ordering 
        [beta, gamma]
    - N: total population size
    - T: end time of the simulation
    - initial_infc: initial number of wild-type infected individuals
    """

    endtime = T 
    initial_SuS = N - initial_infc
    initial_state = [initial_SuS/N, initial_infc/N]
    sdeprob = SIR_SDEProblem(nothing, N, endtime=endtime, initial_state=initial_state)

    return param_list -> SIR_Model_Ensemble_Simulation(sdeprob, param_list)
end

function  SIR_Model_Ensemble_Simulation(sdeprob, param_list)
    """
    Function to create and simulate the EnsembleProblem from a list of parameter vectors.

    Output: A vector of length equal to the number of parameter vectors in param_list.
    Each element of the vector is a 2-dim vector with the solution of the corresponding SDE.
    First dimension is the vector of timepoints, second dimension the vector of state-vectors.
    """
    batchsize = length(param_list)

    # function to change parameters for the different trajectories in the ensemble problem.
    function prob_func(prob, i, repeat)
            params = param_list[i]
            
            remake(prob, p=params)
    end

    function output_func(sol, i)
        return [sol.t, sol.u]
    end

    ensemble_prob = EnsembleProblem(sdeprob, prob_func=prob_func, output_func = output_func, safetycopy=false)
    # solver settings
    # set times where conditions is met, so that the jumpsize is added to the infected individuals

    solve_kwargs = (dt=1e-1, dense=true, force_dtmin=true)
    ensemble_sol = solve(ensemble_prob, PosEM(), trajectories=batchsize; solve_kwargs...)
    return ensemble_sol
end


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

include("../utils/posEM.jl")


# parameters: [gamma_inverse, kappa_inverse, beta, t_event, scaling, initial_infc, variant_infc]

function bind_sbml_params(prob, values::Vector{Float64})
    syms = parameters(prob.f.sys)
    if length(values) != length(syms)
        error("Parameter count mismatch: got $(length(values)), expected $(length(syms))")
    end
    return Dict(zip(syms, values))
end

function sbml_to_SDEProblem(sbml_file, parammap; endtime=1000.0, initial_state=nothing)

    # set the timespan
    tspan = (0., endtime)

    # import the model from SBML
    model = readSBML(sbml_file, doc -> begin
        set_level_and_version(3, 2)(doc)
        convert_promotelocals_expandfuns(doc)
    end)

    # create the reaction system
    rs_sde = ReactionSystem(model)

    # create the SDE-problem form the reaction system
    SDEsys = complete(convert(SDESystem, rs_sde))

    # create the SDE system
    if isnothing(initial_state)
        sdeprob = SDEProblem(SDEsys, [], tspan, parammap)
    else
        sdeprob = SDEProblem(SDEsys, initial_state, tspan, parammap)
    end

    return sdeprob
end

function SEIR_Variant_Model_Simulation(sdeprob, params, solve_alg, solve_kwargs)

    if params[1:4] != sdeprob.p[1:4]
        sdeprob = remake(sdeprob, p=[params..., 1.0])
    end
    sde_sol = solve(sdeprob, solve_alg, solve_kwargs...)

    return sde_sol
end


function SEIR_Variant_SDE(N, T)
    """
    Function to create the SEIR variant model.
    - params: parameter vector of the model in the ordering 
        [gamma_inverse, kappa_inverse, beta, t_event, initial_infc, variant_infc]
    - N: total population size
    - T: end time of the simulation
    - initial_infc: initial number of wild-type infected individuals
    - var_infc: initial number of variant infected individuals introduced at t_event
    """
    model_name = "covid_ethiopia_seir_variant_model_real_pop_current"
    petab_folder = "./petab/virus_variant_model"
    sbml_file = string(petab_folder, "/", model_name, ".sbml")

    endtime = T 

    # set dummy initial state
    initial_infc = N/100
    initial_SuS = N - initial_infc
    initial_state = [0.0, initial_SuS, 0.0, 0.0, initial_infc, 0.0, 0.0, 0.0, 0.0, 0.0 ]
    sdeprob = sbml_to_SDEProblem(sbml_file, [], endtime=endtime, initial_state=initial_state)

    return params -> SEIR_Variant_Model_Simulation(sdeprob, params)
end

function SEIR_Variant_Model_Simulation(sdeprob, params)
    """
    Function to simulate the SEIR variant model. 
    It returns the timepoints and corresponding state-vectors of the system.
    """
    #if params[1:4] != sdeprob.p[1:4]
    #    sdeprob = remake(sdeprob, p=[params[1:4]..., 1.0])
    #end
    params2 = [params[3], params[2], params[1], params[4], 1.0]
    sdeprob = remake(sdeprob; p=bind_sbml_params(sdeprob, params2))
    # get N from the SDEProblem
    N = sum(sdeprob.u0)
    # remake SDEProblem with the new initial infection count
	#N = 180000
    initial_infc = params[5]
    initial_SuS = N - initial_infc
    initial_state = [0.0, initial_SuS, 0.0, 0.0, initial_infc, 0.0, 0.0, 0.0, 0.0, 0.0 ]
    sdeprob = remake(sdeprob, u0=initial_state)

    tevent = params[4]
    jumpsize = params[6]

    make_event_cb(tevent, jumpsize) = DiscreteCallback((u,t,integrator) -> t == tevent, integrator -> integrator.u[10] += jumpsize)
    cb = make_event_cb(tevent, jumpsize)

    # solver settings
    solve_kwargs = (dt=1e-1, callback=cb, tstops=[tevent], dense=true, force_dtmin=true)
    nothing

    sde_sol = solve(sdeprob, PositiveEM(); solve_kwargs...)

    return sde_sol.t, sde_sol.u
end

function SEIR_Variant_SDE_Ensemble(N, T)
    """
    Function to create the SEIR variant model.
    - params: parameter vector of the model in the ordering 
        [gamma_inverse, kappa_inverse, beta, t_event, initial_infc, variant_infc]
    - N: total population size
    - T: end time of the simulation
    - initial_infc: initial number of wild-type infected individuals
    - var_infc: initial number of variant infected individuals introduced at t_event
    """
    model_name = "covid_ethiopia_seir_variant_model_real_pop_current"
    petab_folder = "./petab/virus_variant_model"
    sbml_file = string(petab_folder, "/", model_name, ".sbml")

    endtime = T 
    # set dummy initial state
    initial_infc = N/100
    initial_SuS = N - initial_infc
    initial_state = [0.0, initial_SuS, 0.0, 0.0, initial_infc, 0.0, 0.0, 0.0, 0.0, 0.0 ]
    sdeprob = sbml_to_SDEProblem(sbml_file, [], endtime=endtime, initial_state=initial_state)

    return param_list -> SEIR_Variant_Model_Ensemble_Simulation(sdeprob, N, param_list)
end

function  SEIR_Variant_Model_Ensemble_Simulation(sdeprob, N, param_list)
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

            # remake SDEProblem with the new initial infection count
            initial_infc = params[5]
            initial_SuS = N - initial_infc
            initial_state = [0.0, initial_SuS, 0.0, 0.0, initial_infc, 0.0, 0.0, 0.0, 0.0, 0.0 ]

            tevent = params[4]
            jumpsize = params[6]
            make_event_cb(tevent, jumpsize) = DiscreteCallback((u,t,integrator) -> t == tevent, integrator -> integrator.u[10] += jumpsize)
            cb = make_event_cb(tevent, jumpsize)
            remake(prob, p=[params[1:4]..., 1.0], u0=initial_state, callback = cb)
    end

    function output_func(sol, i)
        return [sol.t, sol.u]
    end

    ensemble_prob = EnsembleProblem(sdeprob, prob_func=prob_func, output_func = output_func, safetycopy=false)
    # solver settings
    # set times where conditions is met, so that the jumpsize is added to the infected individuals
    event_times = [params[4] for params in param_list]
    solve_kwargs = (tstops=event_times, dt=1e-1, dense=true, force_dtmin=true)
    ensemble_sol = solve(ensemble_prob, PositiveEM(), trajectories=batchsize; solve_kwargs...)
    return ensemble_sol
end



using Distributed


# instantiate and precompile environment in all processes
@everywhere begin
  base_path = "Inference-Methods-for-Stochastic-Compartmental-Models"
  using Pkg; Pkg.activate(base_path)
  Pkg.instantiate(); Pkg.precompile()
end


# stuff needed on workers and main
@everywhere begin

    using LinearAlgebra
    using Random
    using StaticArrays
    using Distributions
    using DifferentialEquations
    using StochasticDiffEq
    using Plots
    # using PEtab # not running on the cluster and I think not needed.
    using CSV
    using DataFrames
    using JLD2
    using MCMCChains
    using MCMCChainsStorage
    using HDF5

    using SBML
    using SBMLToolkit
    using Catalyst
    # using ModelingToolkit

    using Particles
    using ParticlesDE
    using StaticDistributions

    include(joinpath(base_pat, "code/epmodels/virus_variant_est_infc.jl"))
    include(joinpath(base_path,"code/epmodels/utils/posEM.jl"))
    include(joinpath(base_path,"code/utils/utilities.jl"))

    # Slurm Job-array if run on cluster
    # task_id_str = get(ENV, "SLURM_ARRAY_TASK_ID", 0)
    # task_id = parse(Int64, task_id_str)

    # set task_id manually if run locally
    task_id = 1
    
    # set hyperparamters for the particle filter
    niter = 50000
    nparticles = 200

    # set noise model
    noise_model ="normal"

    # Getting the petab model
    model_name = "covid_ethiopia_seir_variant_model_real_pop_current"
    petab_folder = joinpath(base_path, "petab/virus_variant_model") 
    sbml_file = string(petab_folder, "/", model_name, ".sbml")

    # set model initial specis sizes
    # species ordering is: [E_w, S, R_var, E_var_w, I_w, I_var_w, E_var, R_var_w, R_w, I_var]
    init_N = 180000
    init_I = 500
    init_var = 100
    initial_state = [0, init_N-init_I, 0, 0, init_I, 0, 0, 0, 0, 0]
    parammap = []

    # set the timespan
    endtime = 400.
    tspan = (0., endtime )


end



# stuff only needed on workers
@everywhere workers() begin

    using PyCall
    pypesto = pyimport("pypesto")
    
    # measurement times from the petab files
    tobs = [262, 308, 304, 328, 343, 353, 242, 273, 304, 333, 363, 393]
    tobs = sort(tobs)
    tobs = unique(tobs);

    # data extracted from petab file
    data = CSV.read(base_path * "/data/seir2v_real_ethiopia.csv", DataFrame) 
    print(data)

    real_data = Vector{Vector{Union{Missing, Float64}}}()

    for ref_t in tobs
        obs_time(time::Int64) = time == ref_t
        global data_point = filter(:T => obs_time, data)
    
        if data_point.missing_prev == [1]
            prev_meas = data_point.seroprevalence[1]
        elseif data_point.missing_prev == [0]
            prev_meas = missing
        end
        
        if data_point.missing_infc == [1]
            infc_meas = data_point.infection_count[1]
        elseif data_point.missing_infc == [0]
            infc_meas = missing
        end

        append!(real_data, [Vector{Union{Missing, Float64}}([prev_meas, infc_meas])])
    end

    real_data = collect(SVector{2, Union{Missing, Float64}}, real_data)

    # augment data with initial observation
    if tobs[1] != SDE_problem.tspan[1]
        real_data = vcat(missing, real_data)
    end;

    # prior distribution
    struct SEIR_Prior end

    function Random.rand(rng::AbstractRNG, d::SEIR_Prior)
        γ_inv = rand(rng, Normal(15.7, 6.7))
        κ_inv = rand(rng, LogNormal(1.63, 0.5))
        β = rand(rng, Uniform(0.0, 1.0))
        t_event = rand(rng, Uniform(120, 360))
        scaling = rand(rng, Uniform(0.1, 10.0))
        I0 = rand(rng, Uniform(10.0, 1000.0))
        while true
            if β*γ_inv > 0.95
                break
            end
            β = rand(rng, Uniform(0.0, 1.0))
            γ_inv = rand(rng, Normal(15.7, 6.7))
        end
        return [γ_inv, κ_inv, β, t_event, scaling, I0]
    end
    Random.rand(d::SEIR_Prior) = rand(Random.default_rng(), d)


    function Distributions.logpdf(::SEIR_Prior, x)
        if x[3]*x[1] > 0.95
            return logpdf(Normal(15.7, 6.7), x[1]) + logpdf(LogNormal(1.63, 0.5), x[2]) + logpdf(Uniform(0.0, 1.0), x[3]) + logpdf(Uniform(120, 360), x[4]) + logpdf(Uniform(0.0, 100), x[5]) + logpdf(Uniform(10.0, 1000), x[6])
        else
            return -Inf
        end
    end

    # include the ParticleFilter Setup
    if noise_model == "binomial"
        print("Binomial noise not yet implemented.")
    elseif noise_model == "normal"
        include("SEIRRealDataFilterSetup.jl")
    end

    llp = log_posterior(nparticles)

    # for pypesto we need the negative log-likelihood
    neg_llp = let llp = llp
        p -> begin
            return -llp(p)
        end
    end

    # transform to pypesto objective
    objective = pypesto.Objective(fun=neg_llp)

    problem = pypesto.Problem(
        objective,
        x_names=["gamma", "kappa", "beta", "tevent", "scaling", "I0"],
        lb=[0,0,0.01,120, 0, 10], # parameter bounds
        ub=[25,25,1,360, 100, 1000], # NB for sampling it is usually better if you remap parameters to (-∞, ∞)
        copy_objective=false, # important
    )

    # specify sampler
    pypesto_sampler = pypesto.sample.AdaptiveMetropolisSampler()

    # sample start value
    x0 = Vector(rand(SEIR_Prior()))
    while true
        global x0 = Vector(rand(SEIR_Prior()));
        if llp(x0) > -200.0 # value might need to be adjusted to get good starting values for the chains
            break
        end
    end

    function chain()
        result = pypesto.sample.sample(
                        problem,
                        n_samples=niter,
                        x0=x0, # starting point
                        sampler=pypesto_sampler,
                        )
        return Chains_from_pypesto(result)
    end

end

jobs = [@spawnat(i, @timed(chain())) for i in workers()]

all_chains = map(fetch, jobs)

chains = all_chains[1].value.value.data

for j in 2:nworkers()
    global chains
    chains = cat(chains, all_chains[j].value.value.data, dims=(3,3))
end

chs = MCMCChains.Chains(chains, [:gamma, :kappa, :beta, :tevent, :scaling, :I0, :lp])
complete_chain = set_section(chs, Dict(:parameters => [:gamma, :kappa, :beta, :tevent, :scaling, :I0], :internals => [:lp]))
stop_time = mean([all_chains[i].time for i in 1:nworkers()])
complete_chain = setinfo(complete_chain, (start_time=1.0, stop_time=stop_time))

print("Mean duration per chain: ", stop_time)

# store results
result_folder = joinpath(basepath, "output/PF_Experiments/seir2v_full_real_ethiopia")

h5open(result_folder * "/real_$(noise_model)_"*string(nworkers())*"chs_"*string(niter)*"it_"*string(nparticles)*"p.h5", "w") do f
  write(f, complete_chain)
end


open(result_folder * "/time_real_$(noise_model)_"*string(nworkers())*"chs_"*string(niter)*"it_"*string(nparticles)*"p.txt", "w") do file
    write(file, stop_time)
end

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


    include(joinpath(base_path,"code/epmodels/utils/posEM.jl"))
    include(joinpath(base_path,"code/utils/utilities.jl"))

    # Slurm Job-array if run on cluster
    task_id_str = get(ENV, "SLURM_ARRAY_TASK_ID", 0)
    task_id = parse(Int64, task_id_str)

    # set task_id manually if run locally
    # task_ id = 1

    # set hyperparamters
    niter = 50000
    nparticles = 200

    # set noise model
    noise_model ="normal"

    # set dataset 
    datasets = ["1_1_1", "1_1_2", "1_1_3", "1_2_1", "1_2_2", "1_2_3"]
    dataset = datasets[task_id+1]

    #set prior
    prior = "normal"
    
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

    SDE_problem = sbml_to_SDEProblem(sbml_file, parammap, endtime=400.0, initial_state=initial_state);

    # parameters are in the ordering [gamma-1, kappa-1, beta, t_event, scaling, I0] in the SDEProblem 
    true_pars = [
        [17, 5, 0.08, 150, 3, 500], 
        [11.7, 8.4, 0.23, 222, 1.6, 560]
        ]
    true_par = true_pars[parse(Int, split(dataset, "_")[1])]

end



# stuff only needed on workers
@everywhere workers() begin


    using PyCall
    pypesto = pyimport("pypesto")

    # set prior
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

    if prior == "uniform"
        function Distributions.logpdf(::SEIR_Prior, x)
            if x[3]*x[1] > 0.95
                return logpdf(Uniform(0.0, 25.0), x[1]) + logpdf(Uniform(0.0, 25.0), x[2]) + logpdf(Uniform(0.0, 1.0), x[3]) + logpdf(Uniform(120, 360), x[4]) + logpdf(Uniform(0.0, 10), x[5]) + logpdf(Uniform(10.0, 1000), x[6])
            else
                return -Inf
            end
        end
    else
        function Distributions.logpdf(::SEIR_Prior, x)
            if x[3]*x[1] > 0.95
                return logpdf(Normal(15.7, 6.7), x[1]) + logpdf(LogNormal(1.63, 0.5), x[2]) + logpdf(Uniform(0.0, 1.0), x[3]) + logpdf(Uniform(120, 360), x[4]) + logpdf(Uniform(0.0, 10), x[5]) + logpdf(Uniform(10.0, 1000), x[6])
            else
                return -Inf
            end
        end
    end

    # observation function for infection cases and prevalence using a normal noise model
    nobs = 2
    # include the ParticleFilter Setup
    if noise_model == "binomial"
        # load data and observation settings
        data_df = CSV.read(base_path * "/data/seir2v_synth_sparse_$dataset.csv", DataFrame) # cluster
        infc_counts = data_df[!, "infection_count"]
	    prev_counts = data_df[!, "Seroprev"]

        # set observation timepoints
        tobs = data_df[!, "timepoint"]

        real_data = Vector{Vector{Union{Missing, Float64}}}()
        for i in range(1, length(tobs))
            if infc_counts[Int64(i)] == -1
            infections_meas = missing
            else
            infections_meas = Int64(500*infc_counts[Int64(i)])
            end
            if prev_counts[Int64(i)] == -1
            prev_meas = missing
            else
            prev_meas = Int64(500*prev_counts[Int64(i)])
            end
            append!(real_data, [Vector{Union{Missing, Float64}}([infections_meas, prev_meas])])
        end

        real_data = collect(SVector{nobs, Union{Float64, Missing}}, real_data)

        # add initial missing
        if tobs[1] != 0.0
            real_data = vcat(missing, real_data)
        end;
        # load likelihood script
        print("Binomial noise model not implemented yet.")
    elseif noise_model == "normal"
        # load data and observation settings
        data_df = CSV.read(base_path * "/data/seir2v_synth_sparse_$dataset.csv", DataFrame) # cluster
        infc_counts = data_df[!, "infection_count"]
        prev_counts = data_df[!, "Seroprev"]
        infc_counts = [x == -1 ? missing : x for x in infc_counts]
        prev_counts = [x == -1 ? missing : x for x in prev_counts]
        real_data = [[infc_counts[i], prev_counts[i]] for i in eachindex(infc_counts)]
        real_data = collect(SVector{nobs, Union{Float64, Missing}}, real_data)

        # set observation timepoints
        tobs = data_df[!, "timepoint"]
        # add initial missing
        if tobs[1] != 0.0
            real_data = vcat(missing, real_data)
        end;

        # observation noise
        noise_infc = data_df[!, "Std"] 
        noise_prev = data_df[!, "Std_1"]

        #load likelihood script
        include("SEIRSparseNormalDataFilterSetup.jl")
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
        ub=[25,25,1,360, 10, 1000], # NB for sampling it is usually better if you remap parameters to (-∞, ∞)
        copy_objective=false, # important
    )

    # specify sampler
    pypesto_sampler = pypesto.sample.AdaptiveMetropolisSampler()

    # sample start value
    x0 = Vector(rand(SEIR_Prior()))
    while true
        global x0 = Vector(rand(SEIR_Prior()));
        if llp(x0) > -100.0
            break
        end
    end
    x0 = [17, 5, 0.08, 150, 3, 500]

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
result_folder = joinpath(basepath, "output/PF_Experiments/seir2v_full_sparse_$(dataset)")

h5open(result_folder * "/sparse_$(dataset)_$(noise_model)_noise_$(prior)_"*string(nworkers())*"chs_"*string(niter)*"it_"*string(nparticles)*"p.h5", "w") do f
  write(f, complete_chain)
end


open(result_folder * "/time_sparse_$(dataset)_$(noise_model)_noise_$(prior)_"*string(nworkers())*"chs_"*string(niter)*"it_"*string(nparticles)*"p.txt", "w") do file
    write(file, stop_time)
end
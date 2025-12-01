using Distributed


# for running on the cluster
# instantiate and precompile environment in all processes
# @everywhere begin
#   using Pkg; Pkg.activate("/home/vincent/nils_vincent_colab")
#   Pkg.instantiate(); Pkg.precompile()
# end

# for running locally
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
    task_id_str = get(ENV, "SLURM_ARRAY_TASK_ID", 0)
    task_id = parse(Int64, task_id_str)

    # set task_id manually if run locally
    # task_ id = 1
    
    # set hyperparamters for the particle filter
    niter = 50000
    nparticles = 200

    # set noise model
    noise_model ="normal"

    # set dataset 
    datasets = ["1_1", "1_2", "2_1", "2_2", "3_1", "3_2"]
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

    # parameters are in the ordering [e0, r0, s0, t, I0] in the SDEProblem 
    true_pars = [
        [1.36, 22.0, 51.0, 150.0, 500.0], 
        [2.69, 20.1, 18.72, 222, 560], 
        [1.6, 11.0, 33.6, 359, 915]]
    true_par = true_pars[parse(Int, split(dataset, "_")[1])]
    
    function original_parameters(p;kappa=5)
        κ_inv = kappa
        γ_inv = p[2]-κ_inv
        β = p[1]/γ_inv
        scaling = p[3]/γ_inv
        t_event = p[4]
        I0 = p[5]
        return [γ_inv, κ_inv, β, t_event, scaling, I0]
    end
    true_or = original_parameters(true_par)
end



# stuff only needed on workers
@everywhere workers() begin

    using PyCall
    pypesto = pyimport("pypesto")

    # prior distribution
    struct SEIR_Reparam_Prior end

    function Random.rand(rng::AbstractRNG, d::SEIR_Reparam_Prior)
        κ_inv = 5
        r0 = rand(rng, Uniform(0.95, 4.0))
        e0 = rand(rng, Uniform(5.0,30.0))
        s0 = rand(rng, Uniform(1.0, 100.0))
        t_event = rand(rng, Uniform(120, 360))
        I0 = rand(rng, Uniform(10.0, 1000.0))
        return [r0, e0, s0, t_event, I0]
    end
    Random.rand(d::SEIR_Reparam_Prior) = rand(Random.default_rng(), d)

    if prior == "uniform"
        function Distributions.logpdf(::SEIR_Reparam_Prior, x)
            return logpdf(Uniform(0.95, 4.0), x[1]) + logpdf(Uniform(5.0,30.0), x[2]) + logpdf(Uniform(1.0, 100.0), x[3]) + logpdf(Uniform(120, 360), x[4]) + logpdf(Uniform(10.0, 1000.0), x[5])
        end
    else
        function Distributions.logpdf(::SEIR_Reparam_Prior, x)
            return logpdf(truncated(Normal(2.2, 0.4), lower=0.95), x[1]) + logpdf(truncated(Normal(20.0,5.0), lower=5.0), x[2]) + logpdf(Uniform(1.0, 100.0), x[3]) + logpdf(Uniform(120, 360), x[4]) + logpdf(Uniform(10.0, 1000.0), x[5])
        end
    end

    # observation function for infection cases and prevalence using a normal noise model
    nobs = 2
    # include the ParticleFilter Setup
    if noise_model == "binomial"
        # load data and observation settings
        data_df = CSV.read(base_path * "/data/seir2v_synth_dense_$dataset.csv", DataFrame) # cluster
        infc_counts = data_df[!, "infection_count"]
	prev_counts = data_df[!, "Seroprev"]

	# set observation timepoints
	tobs = data_df[!, "timepoint"]

	real_data = Vector{Vector{Union{Missing, Float64}}}()
	for i in range(1, length(tobs))
    	    infections_meas = Int64(500*infc_counts[Int64(i)])
    	    prev_meas = Int64(500*prev_counts[Int64(i)])
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
        data_df = CSV.read(base_path * "/data/seir2v_synth_dense_$dataset.csv", DataFrame) # cluster
        infc_counts = data_df[!, "infection_count"]
        prev_counts = data_df[!, "Seroprev"]
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
        include("SEIRReparamDenseNormalDataFilterSetup.jl")
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
        x_names=["r0", "e0", "s0", "tevent", "I0"],
        lb=[0.95, 5.0, 1.0, 120, 10], # parameter bounds
        ub=[4.0, 30.0, 100, 360, 1000], # NB for sampling it is usually better if you remap parameters to (-∞, ∞)
        copy_objective=false, # important
        )

    # specify sampler
    pypesto_sampler = pypesto.sample.AdaptiveMetropolisSampler()

    # sample start value
    x0 = Vector(rand(SEIR_Reparam_Prior()))
    while true
        global x0 = Vector(rand(SEIR_Reparam_Prior()));
        if llp(x0) > -100.0
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

chs = MCMCChains.Chains(chains, [:r0, :e0, :s0, :tevent, :I0, :lp])
complete_chain = set_section(chs, Dict(:parameters => [:r0, :e0, :s0, :tevent, :I0], :internals => [:lp]))
stop_time = mean([all_chains[i].time for i in 1:nworkers()])
complete_chain = setinfo(complete_chain, (start_time=1.0, stop_time=stop_time))

print("Mean duration per chain: ", stop_time)

# store results
result_folder = joinpath(basepath, "output/PF_Experiments/seir2v_reparam_dense_$(dataset)")

h5open(result_folder * "/reparam_dense_$(dataset)_$(noise_model)_noise_$(prior)_"*string(nworkers())*"chs_"*string(niter)*"it_"*string(nparticles)*"p.h5", "w") do f
  write(f, complete_chain)
end


open(result_folder * "/time_reparam_dense_$(dataset)_$(noise_model)_noise_$(prior)_"*string(nworkers())*"chs_"*string(niter)*"it_"*string(nparticles)*"p.txt", "w") do file
    write(file, stop_time)
end

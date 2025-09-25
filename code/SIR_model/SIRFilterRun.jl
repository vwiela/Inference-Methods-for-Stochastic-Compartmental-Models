using Distributed # package for distributed computing in julia


# instantiate and precompile environment in all processes
@everywhere begin
  using Pkg; Pkg.activate(dirname(pwd()))
  Pkg.instantiate(); Pkg.precompile()
end

# stuff needed on workers and main
@everywhere begin   
    using SBMLToolkit, ModelingToolkit, DifferentialEquations, StochasticDiffEq
    using Plots
    using DataFrames
    using CSV
    using Random
    using Distributions
    using SBML
    using SymbolicUtils
    using StaticArrays
    using Catalyst
    using AdvancedMH
    using MCMCChains
    using MCMCChainsStorage
    using StatsPlots
    using ArviZ
    using HDF5

    # Lorenzos packages
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
    nparticles = 100

    # set noise model
    noise_model ="normal"

    # set dataset 
    datasets = ["1", "2"]
    dataset = datasets[task_id+1]

    # SIR-model settings and parameters
    N = 180000
    init_I = 1800
    init_S = N - init_I
    u0 = [init_S/N; init_I/N]

    endtime = 100.0
    tspan = (0.0, endtime)

    # define SDe problem
    SDE_problem = SIR_SDEProblem(nothing, N, endtime=endtime, initial_state=u0);

    # parameters are in the ordering [beta, gamma]
    true_pars = [
        [0.2, 0.05], 
        [0.22, 0.2]]
    true_par = true_pars[parse(Int, dataset)]
end    


# stuff only needed on workers
@everywhere workers() begin
    # get data and define observation model
    nobs = 2
    
    # include the ParticleFilter Setup
    if noise_model == "binomial"
        # load data and observation settings
        data_df = CSV.read(base_path * "/data/sir_dense_normal_and_binomial_$dataset.csv", DataFrame) # cluster
        infc_counts = data_df[!, "infection count"]
        prev_counts = data_df[!, "seroprev"]

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

        # load model
        print("Binomial noise model not implemented yet.")
    elseif noise_model == "normal"
        # load data and observation settings
        data_df = CSV.read(base_path * "/data/sir_dense_normal_and_binomial_$dataset.csv", DataFrame) # cluster
        # print(data_df)
        infc_counts = data_df[!, "infection count"]
        prev_counts = data_df[!, "seroprev"]

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
        noise_prev = data_df[!, "std"]

        # load model
        include("SIRNormalFilterSetup.jl")
    end

    # convert PyPesto result to MCMCChains.jl chain type
    function Chains_from_pypesto(result; kwargs...)
        trace_x = result.sample_result["trace_x"] # parameter values
        trace_neglogp = result.sample_result["trace_neglogpost"] # posterior values
        samples = Array{Float64}(undef, size(trace_x, 2), size(trace_x, 3) + 1, size(trace_x, 1))
        samples[:, begin:end-1, :] .= PermutedDimsArray(trace_x, (2, 3, 1))
        samples[:, end, :] = .-PermutedDimsArray(trace_neglogp, (2, 1))
        param_names = Symbol.(result.problem.x_names)
        chain = Chains(
            samples,
            vcat(param_names, :lp),
            (parameters = param_names, internals = [:lp]);
            kwargs...
        )
        return chain
    end
    
    log_post = log_posterior(nparticles)

    # for pypesto we need the negative log-likelihood
    neg_lp = let log_post = log_post
        p -> begin
            return -log_post(p)
        end
    end

    # transform to pypesto objective
    objective = pypesto.Objective(fun=neg_lp)


    # create pypesto problem

    pypesto_problem = pypesto.Problem(
        objective,
        x_names=["beta", "gamma"],
        lb=[0.001, 0.001], # parameter bounds
        ub=[1, 1], # NB for sampling it is usually better if you remap parameters to (-∞, ∞)
        copy_objective=false, # important
    )

    # specify sampler
    pypesto_sampler = pypesto.sample.AdaptiveMetropolisSampler();

    # get initial parameters
    init_par = rand(SIR_Prior())
        
     # function for sampling and conversion 
    function chain()
        result = pypesto.sample.sample(
                    pypesto_problem,
                    n_samples=niter,
                    x0=Vector(init_par), # starting point
                    sampler=pypesto_sampler,
                    )
       return  Chains_from_pypesto(result)
    end
end

# initialize and run the jobs for the workers
jobs = [@spawnat(i, @timed(chain())) for i in workers()]

all_chains = map(fetch, jobs)

chains = all_chains[1].value.value.data

# get the chains
for j in 2:nworkers()
    global chains
    chains = cat(chains, all_chains[j].value.value.data, dims=(3,3))
end


chs = MCMCChains.Chains(chains, [:beta, :gamma, :lp])
complete_chain = set_section(chs, Dict(:parameters => [:beta, :gamma], :internals => [:lp]))

# get mean computation time per chain
stop_time = mean([all_chains[i].time for i in 1:nworkers()])

# store results
print("Mean runtime for $nparticles particles $niter iterations: ", stop_time)

# store results
result_folder = joinpath(basepath, "output/ParticleFilter")

h5open(result_folder * "/SIR_$(dataset)_"*string(nworkers())*"chs_"*string(niter)*"it_"*string(nparticles)*"p.h5", "w") do f
  write(f, complete_chain)
end


open(result_folder * "/time_SIR_$(dataset)_"*string(nworkers())*"chs_"*string(niter)*"it_"*string(nparticles)*"p.txt", "w") do file
    write(file, stop_time)
end




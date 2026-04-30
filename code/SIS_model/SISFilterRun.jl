using Distributed # package for distributed computing in julia


# instantiate and precompile environment in all processes
@everywhere begin
  base_path = "Inference-Methods-for-Stochastic-Compartmental-Models"
  using Pkg; Pkg.activate(base_path)
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

    include(joinpath(base_path, "code/epmodels/sis_model.jl"))
    include(joinpath(base_path,"code/epmodels/utils/posEM.jl"))
    include(joinpath(base_path,"code/utils/utilities.jl"))
    
    # Slurm Job-array if run on cluster
    # task_id_str = get(ENV, "SLURM_ARRAY_TASK_ID", 0)
    # task_id = parse(Int64, task_id_str)

    # set task_id manually if run locally
    task_id = 1

    # set hyperparamters for the particle filter
    niter = 50000
    nparticles = 100

    # set noise model
    noise_model ="normal"

    # set dataset 
    datasets = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    dataset = datasets[task_id+1]

    # SIR-model settings and parameters
    N = 180000
    init_I = 1800
    init_S = N - init_I
    u0 = [init_S/N; init_I/N]

    endtime = 100.0
    tspan = (0.0, endtime)

    # define SDe problem
    SDE_problem = SIS_SDEProblem(nothing, N, endtime=endtime, initial_state=u0);

    # parameters are in the ordering [beta, gamma-1] in the SDEProblem 
    true_pars = [
    	[0.521311005598119, 1/3.42695512413054], 
    	[0.800353512018983, 1/2.48708107481447],
    	[0.646870392668973, 1/2.25318872554484],
    	[0.545308691480284, 1/2.17255533150824],
    	[0.987497220726763, 1/1.2531102040862],
    	[0.383879690191729, 1/4.09942843298096],
    	[0.333205709306596, 1/4.63577333643937],
    	[0.441617915161218, 1/3.82731769256935],
    	[0.392526138430746, 1/4.35250180575202],
    	[0.703719389887278, 1/2.24387986072133]
    ]
    true_par = true_pars[parse(Int, dataset)]
end    


# stuff only needed on workers
@everywhere workers() begin

    # get data
    nobs = 1

    # load data and observation settings
    data_df = CSV.read(base_path * "/data/SIS/sis_$dataset.csv", DataFrame) # cluster

    infc_counts = data_df[!, "infection_count"]

    # get measurement timepoints
    tobs = data_df[!, "timepoint"]

    real_data = collect(SVector{nobs, Union{Float64, Missing}}, real_data)

    # augment data with initial observation
    if tobs[1] != problem.tspan[1]
        real_data = vcat(missing, real_data)
    end;

    # observation noise
    noise_infc = data_df[!, "Std"] 

    # load model
    include("SISFilterSetup.jl")

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
    init_par = rand(SIS_Prior())
        
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
result_folder = joinpath(basepath, "output/PF_Experiments/SIS/sis_$(dataset)")

h5open(result_folder * "/SIS_$(dataset)_"*string(nworkers())*"chs_"*string(niter)*"it_"*string(nparticles)*"p.h5", "w") do f
  write(f, complete_chain)
end


open(result_folder * "/time_SIS_$(dataset)_"*string(nworkers())*"chs_"*string(niter)*"it_"*string(nparticles)*"p.txt", "w") do file
    write(file, stop_time)
end


# evaluate the runs and store last samples for nils as npy.
include(joinpath(base_path,"code/utils/EvaluateParticleFilter.jl"))

figure_folder = mkpath(result_folder * "/figures")

# set parameter whether to save samples for nils
save_samples = false

# get true parameter dictionary
true_par_dict = Dict(
                :beta => true_par[1],
                :gamma => true_par[2];)
                
if niter > 10000
    burnin = Int(niter-10000)
else
    burnin = Int(niter/10)
end
    
mixed_chain = complete_chain[burnin:end]

# check which chain is converged and remove stuck chains.
nparams = length(names(complete_chain))
stuck_flags = Bool[]
for c in 1:nchains
    chain_slice = mixed_chain[:, :, c]
    all_same = any(abs.(quantile(chain_slice).nt.var"2.5%" - quantile(chain_slice).nt.var"97.5%") .< 1e-10)
    push!(stuck_flags, all_same)
end
if all(stuck_flags)
   error("All chains got stuck! No usable chains remain.")
else
    mixed_chain = mixed_chain[:,:, findall(!, stuck_flags)]
end

if size(mixed_chain, 3) == 0
   error("All chains got stuck! No usable chains remain.")
end

if save_samples
    npzwrite(result_folder * "/sis_$(dataset)_"*string(nworkers())*"chs_"*string(niter)*"it_"*string(nparticles)*"p.npy", mixed_chain.value.data)
end

diagnostic_df = MCMC_diagnostics(mixed_chain; autocorlag=autocorlag)
CSV.write(experiment_folder*"/diagnostics_sis_$(dataset).csv", diagnostic_df)



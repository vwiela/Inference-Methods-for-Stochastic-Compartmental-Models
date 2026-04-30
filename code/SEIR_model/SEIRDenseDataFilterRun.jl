using Distributed

# for running on the cluster
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
    
    using CSV
    using DataFrames
    using JLD2
    using MCMCChains
    using MCMCChainsStorage
    using HDF5
    using NPZ

    using SBML
    using SBMLToolkit
    using Catalyst

    using Particles
    using ParticlesDE
    using StaticDistributions

    include(joinpath(base_pat, "code/epmodels/sir_model.jl"))
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

    # set dataset 
    data_structure = "seir2v_full_dense"
    datasets = ["1-1", "1-2", "2-1", "2-2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
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
        [17, 5, 0.08, 150, 3, 500],
        [11.7, 8.4, 0.23, 222, 1.6, 560], 
        [11.7, 8.4, 0.23, 222, 1.6, 560], 
        [15.4679760959048, 3.96762549129714, 0.0654673610905963, 8.39996128563389, 226.130832895375, 477.343419821419],
        [11.5651008051957, 6.10762399610441, 0.16045988656059, 8.33515376781609, 318.663788904886, 169.636098431087],
        [24.04616013764, 6.63138310737117, 0.042266643818475, 6.45317783091597, 122.861436432086, 789.897181984338],
        [22.2063557301224, 4.37038385224586, 0.0807212846539986, 1.91421028403669, 274.049951514103, 831.153990580889],
        [23.3447867279627, 3.52006354866419, 0.0631048612249476, 6.43376938757196, 329.679952772848, 78.9634897174291],
        [16.6579183916909, 3.34826819530982, 0.069939186491281, 4.94727112652624, 289.793609106831, 604.841083559216],
        [17.9833750413301, 6.59095786277281, 0.143032422653819, 3.55476957162995, 317.186988222796, 859.741707222039],
        [16.7696172208108, 5.57462622032648, 0.178561014114138, 1.17622184822688, 172.369286423695, 31.792202870718],
        [18.6280910303071, 9.92886224872726, 0.180681787374913, 1.06898138665638, 180.038269998219, 157.778698336533],
        [8.51205524253091, 3.80050416048443, 0.173094334769164, 4.66100276373362, 299.804042986972, 327.208829481153]
    ]


    true_par = true_pars[parse(Int, split(dataset, "_")[1])]
    # for the comparison with reparametrized model on same datasets
    # true_pars_comparison = [
    # [8.62787761200484, 5, 0.375393336602948, 138.784181997657, 2.4102852603369, 563.002110927854],
    # [1.83379769205424, 5, 0.661725539272205, 353.890179219662, 18.735364202169,410.501304231619],
    # [12.4769355515914, 5, 0.210402719204872, 252.829191760759, 3.56650339976819, 551.701765710471],
    # [16.430494029293, 5, 0.217055292292094, 338.692304381882, 2.79907789979129, 114.306022189708],
    # [4.61184924258315, 5, 0.267797649237272, 171.070348733889, 11.121576280385, 227.178559309092],
    # [5.7915840251576, 5, 0.21016274588375, 337.676071051568, 0.926929799662752, 497.233166052768],
    # [19.2850506759658, 5, 0.0917217775156078, 141.980152788226, 5.14955742021982, 748.200689682221],
    # [8.70903234281894, 5, 0.121946259625974, 167.766708540823, 9.21429213404834, 940.178217744607],
    # [8.68519433099923, 5, 0.301791523073935, 264.989121791907, 0.928847969446947, 135.078937525541],
    # [22.2234541344768, 5, 0.0524848937409058, 147.515301583979, 1.60622634001999, 428.88520308839]
    # ]
    # true_par = true_pars_comparison[task_id+1]

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
        data_df = CSV.read(base_path * "/data/SEIR2V_full_dense/$(data_structure)_d-$dataset.csv", DataFrame) # cluster
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
        data_df = CSV.read(base_path * "/data/SEIR2V_full_dense/$(data_structure)_d-$dataset.csv", DataFrame) # cluster
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
        include("SEIRDenseNormalDataFilterSetup.jl")
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
        ub=[25,25,1,360, 10, 1000], # NB for sampling it might be better if one remaps parameters to (-∞, ∞)
        copy_objective=false, # important
    )

    # specify sampler
    pypesto_sampler = pypesto.sample.AdaptiveMetropolisSampler()

    # sample start value
    x0 = Vector(rand(SEIR_Prior()))
    while true
        global x0 = Vector(rand(SEIR_Prior()));
        if llp(x0) > -100.0 # for different models this value might need to be adjusted
            break
        end
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
result_folder = joinpath(basepath, "output/PF_Experiments/SEIR2V_full_dense/seir2v_d_$(dataset)")

h5open(result_folder * "/dense_$(dataset)_$(noise_model)_noise_$(prior)_"*string(nworkers())*"chs_"*string(niter)*"it_"*string(nparticles)*"p.h5", "w") do f
  write(f, complete_chain)
end


open(result_folder * "/time_dense_$(dataset)_$(noise_model)_noise_$(prior)_"*string(nworkers())*"chs_"*string(niter)*"it_"*string(nparticles)*"p.txt", "w") do file
    write(file, stop_time)
end


# get true parameter dictionary
true_par_dict = Dict(
                :beta => true_par[3],
                :gamma => true_par[1], 
                :kappa => true_par[2], 
                :tevent => true_par[4],
                :scaling => true_par[5],
                :I0 => true_par[6]);
                
if niter > 10000
    burnin = Int(niter-10000)
else
    burnin = Int(niter/10)
end
    
mixed_chain = complete_chain[burnin:end]


# evaluate the runs and store last samples for further use as npy.
include(joinpath(base_path,"code/utils/EvaluateParticleFilter.jl"))
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

save_samples=false
if save_samples
    npzwrite(result_folder * "/dense_$(dataset)_"*string(nchains)*"chs_"*string(niter)*"it_"*string(nparticles)*"p.npy", mixed_chain.value.data)	
end


diagnostic_df = MCMC_diagnostics(mixed_chain; autocorlag=autocorlag)
CSV.write(result_folder*"/diagnostics_dense_$(dataset).csv", diagnostic_df)

using Random
using Distributions
using StatsFuns: softplus
using Turing
using MCMCChains
using MCMCDiagnosticTools
using CSV
using DataFrames
using ReverseDiff
using HDF5
using MCMCChainsStorage
using NPZ

# Slurm Job-array

task_id_str = get(ENV, "SLURM_ARRAY_TASK_ID", 0)
task_id = parse(Int64, task_id_str)

# set hyperparamters; number of iterations and particles
niter = 10_000
nchains = 4

# set dataset and true pars for the run
datasets = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
dataset = datasets[task_id+1]

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
true_par = true_pars[task_id+1]

# set paths
base_path = "Inference-Methods-for-Stochastic-Compartmental-Models"   # adjust
result_folder = mkpath(base_path * "/output/SIS/sis_$(dataset)")

# -----------------------
# Build a latent time grid that includes observation times
# -----------------------
function make_time_grid_exact(t0::Real, t_obs::AbstractVector{<:Real}; hmax::Real=0.1)
    t_obs = Float64.(t_obs)
    @assert issorted(t_obs) && all(diff(t_obs) .> 0) "t_obs must be strictly increasing."
    @assert t0 < t_obs[1] "t0 must be < first observation time"

    # build from [t0, t_obs...]
    t_all = vcat(Float64(t0), t_obs)

    t_grid = Float64[t_all[1]]
    obs_idx = Int[]  # indices in t_grid corresponding to each t_obs element

    for j in 2:length(t_all)
        tA, tB = t_all[j-1], t_all[j]
        Δ = tB - tA
        n = max(1, ceil(Int, Δ / hmax))
        h = Δ / n

        for k in 1:(n-1)
            push!(t_grid, tA + k*h)
        end
        push!(t_grid, tB)

        # record index if this endpoint is an observation time (j>=2 means it is)
        if j >= 2
            push!(obs_idx, length(t_grid))
        end
    end

    @assert all(diff(t_grid) .> 0)
    return t_grid, obs_idx
end

# Smoothly map (s,i) to positive fractions summing to 1 for rate calculations
# (helps avoid sqrt of negative, while keeping the 2D SDE form).

clamp01(x; eps=1e-12) = min(max(x, eps), 1 - eps)

# -----------------------
# 2D SIS SDE (your drift + your diffusion matrix), EM discretization
# -----------------------

@model function sis_sde_2d_one_innovation(
    yI,
    t_grid::Vector{Float64},
    obs_idx::Vector{Int},
    infc_noise::Vector{Float64};
    N::Int=180_000
)
    β ~ Uniform(0.2, 1.)
    γ ~ Uniform(0.05, 1)
#     R0 ~ Uniform(1.1, 2.5)
#     β = R0 * γ

    Tβ = typeof(β)

    i0 = 1800.0 / N # initial fraction of infected
    i_prev = clamp01(i0)
    s_prev = one(Tβ) - i_prev # initial fraction of susceptible

    K = length(t_grid)

    η ~ filldist(Normal(), K-1)    
    m = length(obs_idx)
    i_at_obs = Vector{typeof(β)}(undef, m)
    obs_cursor = 1

    # record at k=1 if it is an observation index (it won’t be if obs start at 5)
    if obs_cursor ≤ m && 1 == obs_idx[obs_cursor]
        i_at_obs[obs_cursor] = i_prev
        obs_cursor += 1
    end
    
    s_at_obs = Vector{Tβ}(undef, length(obs_idx))
    i_at_obs = Vector{Tβ}(undef, length(obs_idx))
    s_at_obs[1] = s_prev
    i_at_obs[1] = i_prev
    obs_cursor = 2

    for k in 2:K
        dt  = t_grid[k] - t_grid[k-1]
        √dt = sqrt(dt)

        i = clamp01(i_prev)
        s = one(Tβ) - i

        ds_drift = (-β*s*i + γ*i)
        di_drift = ( β*s*i - γ*i)

        # a^2 + b^2 variance combo
        var_xi = (β*s*i + γ*i) / N + one(Tβ)*1e-12
        # var_xi  = softpos(var_raw; κ=50.0) + one(Tβ)*1e-12   # always > 0
        σxi = sqrt(var_xi)

        ξ = σxi * √dt * η[k-1]

        s_next = s_prev + ds_drift*dt + ξ
        i_next = i_prev + di_drift*dt - ξ

        # --- project state after step (prevents drift out of [0,1]) ---
        i_prev = clamp01(i_next)
        s_prev = one(Tβ) - i_prev

        if obs_cursor ≤ m && k == obs_idx[obs_cursor]
            i_at_obs[obs_cursor] = i_prev
            obs_cursor += 1
        end
    end

    for j in 1:length(obs_idx)
        yI[j] ~ Normal(i_at_obs[j], infc_noise[j])
    end
end
# -----------------------
# Runner + save + quick diagnostics
# -----------------------

data_df = CSV.read(base_path * "/data/SIS/sis_$dataset.csv", DataFrame) # cluster

infc_counts = data_df[!, "infection_count"]

# get measurement timepoints
t_obs = data_df[!, "timepoint"]

# observation noise
noise_infc = data_df[!, "Std"] 

# emulate your PositiveEM dt≈1e-1 by setting hmax=0.1
hmax = 0.1
t_grid, obs_idx = make_time_grid_exact(0.0, t_obs; hmax=0.1)

N = 180_000
initial_params = [true_par]
model = sis_sde_2d_one_innovation(infc_counts, t_grid, obs_idx, noise_infc; N=N)

# multi-chain sampling

sampler = NUTS(2_000, 0.8; adtype=AutoReverseDiff())
chn = sample(model, sampler, MCMCThreads(), niter, nchains; discard_adapt=true)

# save 
outdir = mkpath(experiment_folder * "/output/HMC_results")
mkpath(outdir)

par_chn = chn[[:β, :γ]]

# quick evaluation
println("\n== summarystats ==")
println(summarystats(par_chn))

println("\n== R̂ ==")
println(rhat(par_chn))

println("\n== ESS ==")
println(ess(par_chn))

# store chain and metadata for later analysis
h5open(outdir * "/HMC_SIS_"*string(nchains)*"chs_"*string(niter)*"it.h5", "w") do f
  write(f, par_chn)
end

npzwrite(outdir * "/HMC_SIS_$(dataset)_"*string(nchains)*"chs_"*string(niter)*"it_"*string(nparticles)*"p.npy", par_chn.value.data)	

using Random
using Distributions
using StatsFuns: softplus
using Turing
using MCMCChains
using MCMCDiagnosticTools
using MCMCChainsStorage
using CSV
using DataFrames
using ReverseDiff
using HDF5
using NPZ

# -----------------------
# Slurm Job-array
# -----------------------
task_id_str = get(ENV, "SLURM_ARRAY_TASK_ID", "0")
task_id = parse(Int64, task_id_str)

niter = 10_000
nchains = 6

datasets = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

dataset = datasets[task_id+1]

# parameters are in the ordering [beta, gamma-1] in the SDEProblem 
true_pars = [
    [0.2, 0.05], 
    [0.22, 0.2],
    [0.430627818147511, 1/7.30774847648868],
    [0.334587638232734, 1/11.0538305689998],
    [0.332865045863081, 1/10.0519095319788],
    [0.496227847255627, 1/8.29271878472793],
    [0.0444706605366109, 1/29.304886912704],
    [0.587056974440183, 1/7.12626648556889],
    [0.392045841507879, 1/9.93678964409357],
    [0.437955267549603, 1/8.50514341810628],
    [0.219283474759941, 1/14.7522040091955],
    [0.912100668563262, 1/1.23966200311736]
    ]
true_par = true_pars[parse(Int, dataset)]

# set paths
base_path = "Inference-Methods-for-Stochastic-Compartmental-Models"   # adjust
result_folder = mkpath(base_path * "/output/PF_Experiments/SIR/sir_$(dataset)")

# -----------------------
# Time grid
# -----------------------
function make_time_grid_exact(t0::Real, t_obs::AbstractVector{<:Real}; hmax::Real=0.1)
    t_obs = Float64.(t_obs)
    @assert issorted(t_obs) && all(diff(t_obs) .> 0) "t_obs must be strictly increasing."
    @assert t0 < t_obs[1] "t0 must be < first observation time"

    t_all = vcat(Float64(t0), t_obs)

    t_grid = Float64[t_all[1]]
    obs_idx = Int[]

    for j in 2:length(t_all)
        tA, tB = t_all[j-1], t_all[j]
        Δ = tB - tA
        n = max(1, ceil(Int, Δ / hmax))
        h = Δ / n

        for k in 1:(n-1)
            push!(t_grid, tA + k*h)
        end
        push!(t_grid, tB)

        # endpoints correspond to observation times for j>=2
        push!(obs_idx, length(t_grid))
    end

    @assert all(diff(t_grid) .> 0)
    return t_grid, obs_idx
end

clamp01(x; eps=1e-12) = min(max(x, eps), 1 - eps)

# -----------------------
# SIR CLE / Euler–Maruyama with TWO reaction noises (infection + recovery)
# Observations:
#   yI[j]      ~ Normal(i(t_j), σI[j] or σI)
#   ySero[j]   ~ Normal(i(t_j)+r(t_j), σSero[j] or σSero)
# -----------------------
@model function sir_sde_3d_two_noises(
    yI,
    ySero,
    t_grid::Vector{Float64},
    obs_idx::Vector{Int};
    N::Int=180_000,
    i0_frac::Float64=1800.0/180_000,
    r0_frac::Float64=0.0,
    noiseI::Union{Nothing,Vector{Float64}}=nothing,
    noiseSero::Union{Nothing,Vector{Float64}}=nothing
)
    # priors (tune to your needs)
    β ~ Uniform(0.2, 1.0)
    γ ~ Uniform(0.05, 1.0)

    # if no noise vectors provided, infer constant sigmas
    if noiseI === nothing
        σI ~ truncated(Cauchy(0, 0.05), 0, Inf)       # fraction scale
    end
    if noiseSero === nothing
        σSero ~ truncated(Cauchy(0, 0.05), 0, Inf)
    end

    Tβ = typeof(β)

    # initial state at t=t_grid[1] (should be 0.0)
    i_prev = clamp01(Tβ(i0_frac))
    r_prev = clamp01(Tβ(r0_frac))
    s_prev = clamp01(one(Tβ) - i_prev - r_prev)

    K = length(t_grid)
    m = length(obs_idx)

    # two innovation sequences (per step)
    η_inf ~ filldist(Normal(), K-1)
    η_rec ~ filldist(Normal(), K-1)

    i_at_obs   = Vector{Tβ}(undef, m)
    ir_at_obs  = Vector{Tβ}(undef, m)  # i+r = seroprev latent
    obs_cursor = 1

    for k in 2:K
        dt  = max(t_grid[k] - t_grid[k-1], 1e-12)
        √dt = sqrt(dt)

        # use clamped state for rates
        i = clamp01(i_prev)
        r = clamp01(r_prev)
        s = clamp01(one(Tβ) - i - r)

        # rates (fraction form)
        rate_inf = β * s * i
        rate_rec = γ * i

        # drift
        ds_drift = -rate_inf
        di_drift =  rate_inf - rate_rec
        dr_drift =  rate_rec

        # CLE noise stds (divide by sqrt(N))
        σ_inf = sqrt(rate_inf / N + one(Tβ)*1e-12)
        σ_rec = sqrt(rate_rec / N + one(Tβ)*1e-12)

        ξ_inf = σ_inf * √dt * η_inf[k-1]
        ξ_rec = σ_rec * √dt * η_rec[k-1]

        # EM update
        s_next = s_prev + ds_drift*dt - ξ_inf
        i_next = i_prev + di_drift*dt + ξ_inf - ξ_rec
        r_next = r_prev + dr_drift*dt + ξ_rec

        # project back (simplex-ish)
        i_prev = clamp01(i_next)
        r_prev = clamp01(r_next)
        s_prev = clamp01(one(Tβ) - i_prev - r_prev)

        if obs_cursor ≤ m && k == obs_idx[obs_cursor]
            i_at_obs[obs_cursor]  = i_prev
            ir_at_obs[obs_cursor] = clamp01(i_prev + r_prev)
            obs_cursor += 1
        end
    end

    # likelihood
    @assert length(yI) == m && length(ySero) == m

    for j in 1:m
        if noiseI === nothing
            yI[j] ~ Normal(i_at_obs[j], σI)
        else
            yI[j] ~ Normal(i_at_obs[j], noiseI[j])
        end

        if noiseSero === nothing
            ySero[j] ~ Normal(ir_at_obs[j], σSero)
        else
            ySero[j] ~ Normal(ir_at_obs[j], noiseSero[j])
        end
    end
end

# -----------------------
# Runner
# -----------------------
Random.seed!(1)

data_df = CSV.read(base_path * "/data/SIR/sir_$(dataset).csv", DataFrame)  # adjust filename

yI = Float64.(data_df[!, "infection_count"])  # fractions
ySero = Float64.(data_df[!, "Seroprev"])      # fractions of I+R
# observation noise
noiseI = data_df[!, "std_1"] 
noiseSero = data_df[!, "std_2"]

t_obs = Float64.(data_df[!, "timepoint"])

hmax = 0.1
t_grid, obs_idx = make_time_grid_exact(0.0, t_obs; hmax=hmax)

N = 180_000

model = sir_sde_3d_two_noises(yI, ySero, t_grid, obs_idx; N=N, i0_frac=1800.0/N, r0_frac=0.0,
                             noiseI=noiseI, noiseSero=noiseSero)

sampler = NUTS(2_000, 0.8; adtype=AutoReverseDiff())
chn = sample(model, sampler, MCMCThreads(), niter, nchains; discard_adapt=true)

# parameters to store
# if noise vectors were not provided, include σI/σSero too
keep_syms = Symbol[:β, :γ]
if noiseI === nothing
    push!(keep_syms, :σI)
end
if noiseSero === nothing
    push!(keep_syms, :σSero)
end
par_chn = chn[keep_syms]

println("\n== summarystats ==")
println(summarystats(par_chn))
println("\n== R̂ ==")
println(rhat(par_chn))
println("\n== ESS ==")
println(ess(par_chn))

outdir = mkpath(result_folder * "/HMC_results")

h5open(outdir * "/HMC_SIR_"*string(nchains)*"chs_"*string(niter)*"it"*string(dataset)*".h5", "w") do f
    write(f, par_chn)
end

npzwrite(outdir * "/HMC_SIR_$(dataset)_"*string(nchains)*"chs_"*string(niter)*"it.npy",par_chn.value.data)

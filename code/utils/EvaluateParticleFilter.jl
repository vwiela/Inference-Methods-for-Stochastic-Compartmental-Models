using Plots
using StatsPlots
using CSV
using DataFrames
using JLD2
using MCMCChains
using MCMCChainsStorage
using HDF5
using LinearAlgebra
using StatsFuns
using OrderedCollections


function original_parameters(p;kappa=5)
    κ_inv = kappa
    γ_inv = p[2]-κ_inv
    β = p[1]/γ_inv
    scaling = p[3]/γ_inv
    t_event = p[4]
    I0 = p[5]
    return [γ_inv, κ_inv, β, t_event, scaling, I0]
end

# transfrom from parameter interval to real line
function transform_single_par(p, lb, ub)
    return logit((p-lb)/(ub-lb))
end

function transform_pars(p, lb, ub)
    return [transform_single_par(p[i], lb[i], ub[i]) for i in eachindex(p)]
end

# re transform from real line to parameter interval
function re_transform_single_par(p, lb, ub)
    return lb+(ub-lb)*logistic(p)
end

function re_transform_pars(p, lb, ub)
    return [re_transform_single_par(p[i], lb[i], ub[i]) for i in eachindex(p)]
end

function retransform_chain(chain, lb, ub)
    samples = hcat(mapslices(x -> re_transform_pars(x,lb,ub), chain[chain.name_map.parameters].value.data; dims=2), reshape(chain[:lp].data, insert!([size(chain[:lp].data)...],2,1)...))
    param_names = chain.name_map.parameters
    Chains(
        samples,
        vcat(param_names, :lp),
        (parameters = param_names, internals = [:lp])
    )
end

function visualize_chain(chain, save_path ; re_transform=false, true_par_dict=nothing)
    niter = size(chain,1)
    nparams = size(chain,2)-1
    nchains = size(chain,3)
    burn_in = Int(round(niter/2))

    if re_transform
        lb = [0,0,0.01,120, 0, 10]
        ub = [25,12,1,360, 100, 1000]
        chain = retransform_chain(chain, lb, ub)
    end

    chain = chain[burn_in:end]

    param_names = chain.name_map.parameters
    # create mean parameter dict
    mean_par_dict = Dict(mean(chain).nt.parameters .=> mean(chain).nt.mean)

    for par in param_names
        plt = density(chain[par], grid=false, widen=false, title="$(String(par))", label="density")
        vline!(plt, [mean_par_dict[par]], color="red", label="Mean value")
        if !isnothing(true_par_dict)
            vline!(plt, [true_par_dict[par]], color="black", label="True value")
        end

        savefig(plt, "$(save_path)_$(String(par))_density.png")
    end

    plt = StatsPlots.plot(chain,  seriestype=:traceplot)

    savefig(plt, "$(save_path)_traceplot.png")

end

function mode_of_chain(chain; return_dict=false)
    mode_vec = []
    for i in eachindex(chain.name_map.parameters)
        par_name = chain.name_map.parameters[i]
        par_kde = kde(vec(MCMCChains.group(chain, par_name).value.data))
        mode = [par_kde.x[argmax(par_kde.density)]]
        push!(mode_vec, mode[1])
    end
    if return_dict
        return OrderedDict(chain.name_map.parameters .=> mode_vec)
    end
    return mode_vec
end

function Root_MSE_from_single_sim(sim, data; binomial=false)
    if binomial
        vecs = [(sim./500 .- data)[i].^2 for i in eachindex(data)]
        return [mean(skipmissing(v[1] for v in vecs)), mean(skipmissing(v[2] for v in vecs))]
    end
    vecs = [(sim .- data)[i].^2 for i in eachindex(data)]
    return [sqrt(mean(skipmissing(v[1] for v in vecs))), sqrt(mean(skipmissing(v[2] for v in vecs)))]
end

function Root_MSE_from_MAP(chain, ssm, data, tobs, make_event_cb; n_sim=200, binomial=false, return_dict=false, reparametrized=false, init_var=100)
    init_N = sum(ssm.u0)
    map = mode_of_chain(chain)
    if reparametrized
        scaled_map = copy(map)
        map = original_parameters(map)
    end
    sde_par = map[[3,4,2,6]]
    u0 = [0, init_N-map[1], 0, 0, map[1], 0, 0, 0, 0, 0]
    scaling = map[5]
    ssm.fobs_kwargs = (; scaling)
    jumpsize = Float64(init_var)
    tevent = Float64(map[6])
    ssm.solve_kwargs = (dt = 1e-1, force_dtmin=true, callback = make_event_cb(tevent, jumpsize), tstops = [tevent]) # NB must be of the same type (order included) as the template passed to SDEStateSpaceModel  # dt=1e-2
    scaling = map[5]
    ssm.fobs_kwargs = (; scaling)

    sde_prob = remake(ssm.sprob, u0=u0, p=[sde_par..., 1.0])
    output_func = (sol, i) -> (rand.(ssm.fobs.(sol(tobs).u, zeros(length(tobs)), tobs; ssm.fobs_kwargs...)), false)

    ens_prob = EnsembleProblem(sde_prob, output_func=output_func)
    ens_sim = solve(ens_prob, ssm.algorithm, EnsembleThreads(), trajectories=n_sim; ssm.solve_kwargs...)
    single_mse = Root_MSE_from_single_sim.(ens_sim.u, [data[.!ismissing.(data)] for _ in 1:n_sim]; binomial=binomial)
    if return_dict 
        map = mode_of_chain(chain, return_dict=true)
        return merge(OrderedDict("MSE_infected" => mean(v[1] for v in single_mse), "MSE_prevalence" => mean(v[2] for v in single_mse)), map)
    end
    return [mean(v[1] for v in single_mse), mean(v[2] for v in single_mse)]
end

function MCMC_diagnostics(chain; autocorlag=100)
    ess_values = ess(chain, kind=:basic)
    gelman_diag = gelmandiag(chain)
    rhat_values = rhat(chain)
    geweke_values = gewekediag(chain)
    ess_values = ess(chain, kind=:basic, maxlag=autocorlag)

    # return as data frame 
    diagnostics_df = DataFrame(parameter=chain.name_map.parameters, 
    rhat=rhat_values.nt.rhat, gelman=gelman_diag.nt.psrf, ess_basic=ess_values.nt.ess)
    for (i, gewe_arr) in enumerate(geweke_values)
        diagnostics_df[!, "Geweke_chain_$i"] = gewe_arr.nt.pvalue
    end
    
    return diagnostics_df
end
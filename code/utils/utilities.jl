using MCMCChains
using Plots
using Distances
using DataFrames
using StatsPlots

# plot genealogical tree of the ancestor history for a particle filter run
function plot_ancestor_tree(ancestors)
    nparticles = length(ancestors[1])
    nobs = length(ancestors)
    lines = Array{Float64}(undef, nparticles, nobs+1);
    B = collect(Int, axes(last(ancestors), 1));
    lines[:,1] = B
    for j in 1:nobs
        B.= getindex.(Ref(ancestors[end-(j-1)]), B)
        lines[:,j+1] = B
    end
    p1 = Plots.plot(1:(nobs+1), reverse(lines[1,:]),color=:black, leg=false)
    for i in 2:nparticles
        Plots.plot!(p1, 1:(nobs+1), reverse(lines[i,:]), color=:black, xlabel="", ylabel="", ticks=false)
    end
    return p1
end

# calculate the ancestors of a particle fitler run; input should be pf.history_pf.ancestors if pf is a instance of a particle filter.
function unique_ancestors_at_previous_times(ancestors::AbstractVector{<:AbstractVector{Int}}; check::Bool=true)
    # NB thanks to the fact that ancestor indices are always ordered,
    #    we only need the history for the first and last particle
    !check || all(issorted, ancestors) || error("ancestors do not satisfy ordering assumptions")
    unique = Vector{Int}(undef, length(ancestors) + 1)
    k = length(ancestors) + 1
    t = lastindex(ancestors)::Int
    B = collect(Int, axes(last(ancestors), 1))
    @inbounds while true
        unique[k] = length(Set(B))
        t ≥ firstindex(ancestors) || break
        B .= getindex.(Ref(ancestors[t]), B)
        t -= 1
        k -= 1
    end
    return unique
end



function logp_vs_nparticles(ssm::StateSpaceModel, data, nparticles::AbstractVector{<:Integer}, theta; nruns::Integer=10, kwargs...)
    x = Vector{String}(undef, length(nparticles) * nruns)
    y = Vector{Float64}(undef, length(nparticles) * nruns)
    k = 1
    for n in nparticles
        logp = LogPosterior(ssm, data, n)
        for _ in 1:nruns
            @inbounds x[k] = string(convert(Int, n))
            @inbounds y[k] = logp(theta)
            k += 1
        end
    end
    plt_df = DataFrame(nparticles=x, logp=y)
    display(@df plt_df violin(:nparticles, :logp))
    return plt_df
end

struct LogPosterior{T_SMC <: SMC, T_CACHE}
    pf::T_SMC
    cache::T_CACHE
    function LogPosterior(ssm::StateSpaceModel, data, nparticles::Integer)
        bf = BootstrapFilter(ssm, data)
        pf = SMC(
            bf, Particles.parameter_template(ssm), nparticles,
            ParticleHistoryLength(; logCnorm=StaticFiniteHistory{1}()),
            NamedTuple(),
            AdaptiveResampling(SystematicResampling(), 0.5),
        )
        cache = Particles.SMCCache(pf)
        return new{typeof(pf), typeof(cache)}(pf, cache)
    end
end

function (logp::LogPosterior)(theta) #::Float64
    reset!(logp.pf, theta)
    offlinefilter!(logp.pf, logp.cache)
    if isnan(logp.pf.history_pf.logCnorm[end])
        println("NaN posterior")
        return logp.pf
    end
    return logp.pf.history_pf.logCnorm[end]
end

struct LogPosteriorWithGradient{T_SMC1 <: SMC, T_SMC2 <: SMC, T_CACHE1, T_CACHE2}
    pf::T_SMC1
    pf_grad::T_SMC2
    cache::T_CACHE1
    cache_grad::T_CACHE2
    function LogPosteriorWithGradient(ssm::StateSpaceModel, data, nparticles::Integer)
        bf = BootstrapFilter(ssm, data)
        pf = SMC(
            bf, Particles.parameter_template(ssm), nparticles,
            ParticleHistoryLength(; logCnorm=StaticFiniteHistory{1}()),
            NamedTuple(),
            AdaptiveResampling(SystematicResampling(), 0.5),
        )
        cache = Particles.SMCCache(pf)
        pf_grad = SMC(
            bf, Particles.parameter_template(ssm), nparticles,
            ParticleHistoryLength(; logCnorm=StaticFiniteHistory{1}()),
            (score=Score(), ),
            AdaptiveResampling(SystematicResampling(), 0.5),
        )
        cache_grad = Particles.SMCCache(pf_grad)
        return new{typeof(pf), typeof(pf_grad), typeof(cache), typeof(cache_grad)}(pf, pf_grad, cache, cache_grad)
    end
end
function (logp::LogPosteriorWithGradient)(gradient, theta)
    reset!(logp.pf_grad, theta)
    offlinefilter!(logp.pf_grad, logp.cache_grad)
    compute_summary!(gradient, logp.pf_grad, :score)
    y = logp.pf_grad.history_pf.logCnorm[end]::Float64
    # if !isfinite(y) || !all(isfinite, gradient)
    #     @info "logp is not finite" theta logp=y ∇logp=gradient
    #     flush(stdout)
    # end
    return y
end
function (logp::LogPosteriorWithGradient)(::Val{:return}, theta)
    gradient = similar(theta, Float64)
    y = logp(gradient, theta)
    return y::Float64, gradient
end
function (logp::LogPosteriorWithGradient)(theta)
    reset!(logp.pf, theta)
    offlinefilter!(logp.pf, logp.cache)
    y = logp.pf.history_pf.logCnorm[end]::Float64
    # if !isfinite(y)
    #     @info "logp is not finite" theta logp=y
    #     flush(stdout)
    # end
    return y
end

# plot coverage of the fitlering distributions from a Bootstrap Fitler run
function my_plot_filter(ssm::StateSpaceModel, parameters, components::AbstractVector{<:Integer}, 
        tobs; nparticles::Integer=500, nsigmas::Real=2, width_px::Integer=800
    )
    colors = palette(:Accent_8)
    length(components) > length(colors) && error("too many components, not enough colors")
    ncomp = length(components)
    hidden, obs=rand(ssm, parameters, length(tobs));
    bf = BootstrapFilter(ssm, obs)
    pf = SMC(
    bf, parameters, nparticles,
    (filter=RunningSummary(MeanAndVariance(), FullHistory()), ),
    )
    offlinefilter!(pf);
    hist = pf.history_run.filter
    comp_means = Vector{Vector{Float64}}(undef, ncomp)
    comp_variances = Vector{Vector{Float64}}(undef, ncomp)
    for i in 1:ncomp
        mean = [hist[j].mean[components[i]] for j in 1:length(tobs)]
        variance = [hist[j].var[components[i]] for j in 1:length(tobs)]
        comp_means[i] = mean
        comp_variances[i] = variance
    end
    plt = Plots.plot(size=(width_px/2,width_px/3))
    x = 1:2:2*length(tobs)
    for i in 1:ncomp        
        plot!(plt, tobs, comp_means[i], color=colors[i])
        plot!(plt, tobs, comp_means[i]+nsigmas*sqrt.(comp_variances[i]),
        fillrange=comp_means[i]-nsigmas*sqrt.(comp_variances[i]),
        alpha=0.35, color=colors[i], label="")
        scatter!(plt, tobs, [hidden[j][components[i]] for j in 1:length(tobs)], color=colors[i], label="")
    end
    return plt
end;


# functions to evalaute the chains



# function to transform pypesto chains to MCMCChains
function Chains_from_pypesto(result; kwargs...)
    trace_x = result.sample_result["trace_x"] # parameter values
    trace_neglogp = result.sample_result["trace_neglogpost"] # posterior values
    samples = Array{Float64}(undef, size(trace_x, 2), size(trace_x, 3) + 1, size(trace_x, 1))
    samples[:, begin:end-1, :] .= PermutedDimsArray(trace_x, (2, 3, 1))
    samples[:, end, :] = .-PermutedDimsArray(trace_neglogp, (2, 1))
    param_names = Symbol.(result.problem.x_names)
    return Chains(
        samples,
        vcat(param_names, :lp),
        (parameters = param_names, internals = [:lp]);
        kwargs...
    )
end



# compute the burn_in
function burn_in_from_geweke(chain::Chains, z_threshold::Float64=2.0)
    niter = size(chain)[1]
    npar = size(chain)[2]-1 #-1 because of the internal lp
    nchains = size(chain)[3]
    # number of fragments
    n = 10
    step = Int(floor(niter/n))
    fragments = 0:step:niter-20
    z = zeros(length(fragments), npar, nchains)
    burn_in_list = []
    for j in 1:nchains
        for (i, indices) in enumerate(fragments)
            z[i, :, j] = DataFrame(gewekediag(chain[indices+1:end,:, j]))[!,"zscore"][1]
        end
        max_z = maximum(abs.(z[:,:,j]), dims=2) #note that it returns a matrix with one column
        idxs = sortperm(max_z[:,1], rev=true) #sort descending
        alpha2 = z_threshold * ones(length(idxs))
        max_z = maximum(abs.(z[:,:,j]), dims=2)
        idxs = sortperm(max_z[:,1], rev=true)
        alpha2 = z_threshold * ones(length(idxs))
        for k in 1:length(max_z)
           alpha2[idxs[k]] = alpha2[idxs[k]]/(length(fragments)-findfirst(==(k), idxs) +1) 
        end
        if any(alpha2.>max_z)
            burn_in = findfirst((alpha2 .> max_z)[:,1]) * step
        else
            burn_in = niter
        end
        append!(burn_in_list, burn_in)
    end
    return Int64(maximum(burn_in_list)) #a conservative choice is the maximum of all chains; or median for a less conservative choice
end

# compute the mean-squared jumping distance
function mean_squared_jumping_distance(chain, parameter_names::Vector{String}; burn_in::Integer=0)
    burnin = burn_in+1
    niter = size(chain)[1]
    nchains = size(chain)[3]
    names = cat(parameter_names, "combined", dims=1)
    msjd_df = DataFrame()
    for idx_chain in 1:nchains
        df = DataFrame()
        for p in parameter_names
            m = mean([(chain[i,p,idx_chain] - chain[i-1,p,idx_chain])^2 for i in (burnin+1):niter])
            df[!,p] = [m]
        end
        m_c = mean([euclidean(chain.value[i,:,idx_chain][1:length(parameter_names)],chain.value[i-1,:,idx_chain][1:length(parameter_names)])^2 for i in (burnin+1):niter])
        df[!,"total_msjd"] = [m_c]
        append!(msjd_df, df)
    end
    return msjd_df
end

# compute mean acceptance rate
function acceptance_rate(chain; burn_in::Integer=0)
    burnin = burn_in+1
    nchains=size(chain)[3]
    niter = size(chain)[1]
    acc_rate_df = DataFrame(acceptance_rate=[]) 
    for idx_chain in 1:nchains
        rate = mean([chain.value[i,:,idx_chain] == chain.value[i+1,:,idx_chain] ? 1 : 0 for i in burnin:niter-1])
        push!(acc_rate_df, [rate])
   end
   return acc_rate_df
end
function evaluate_synth_chain(chain, parameter_names)
    burn_in = burn_in_from_geweke(chain)
    if burn_in==size(chain)[1]
        print("Geweke test indicates no convergence of the chain")
        burn_in = Int(round(size(chain)[1]/2))
        # return nothing
    end
    print("Burn In: ", burn_in, "\n")
    msjd = DataFrame(MSJD=mean_squared_jumping_distance(chain, parameter_names;burn_in=burn_in).total_msjd)
    print(hcat(msjd,acceptance_rate(chain; burn_in=burn_in)))
    print("\n", DataFrame(summarize(chain[burn_in:end])))
    display(StatsPlots.plot(chain[burn_in:end]))
    return nothing
end


# check the coverage of the true data by the SDE-model with the given parameters
function check_function(SDE_problem, par_new, tobs, y_load) 
    

    prob_func = let SDE_problem = SDE_problem
        (prob, i, repeat) -> begin
            u0_new = copy(SDE_problem.u0)
            parameter = copy(SDE_problem.p)
            parameter[1:length(par_new)] .= par_new
            return remake(SDE_problem; u0=u0_new, p=parameter)
        end
    end

    # solver settings
    solve_alg = PosEM()
    solve_kwargs = (dt=1e-2,)
    nothing
    
    # create ensemble problem and simulate it 500 tiems
    output_func = (sol, i) -> ([rand(fobs(sol(t), nothing, t)) for t in tobs], false)
    ensemble_prob = EnsembleProblem(SDE_problem; output_func, prob_func)
    sim = solve(ensemble_prob, solve_alg; trajectories=500, solve_kwargs...);
    
    # get median and std. errors of the simulations
    Prev = Matrix{Float64}(undef, length(sim), length(tobs))
    Infc = Matrix{Float64}(undef, length(sim), length(tobs))
    for i in range(1,length(sim))
        prevalence = []
        infections = []
        for j in range(1, length(tobs))
            push!(prevalence, sim.u[i][j][1])
            push!(infections, sim.u[i][j][2])
        end
        Prev[i,:] = prevalence
        Infc[i,:] = infections
    end
    return mean(Prev; dims=1), median(Infc; dims=1), std(Prev;dims=1), std(Infc;dims=1)
end

# plot the data coverage ability of the SDE prolem with the new parameter
function data_check_plot(SDE_problem, par_new, tobs, y_load)
    Prev_mean, Infc_mean, Prev_std, Infc_std = check_function(problem, true_params, tobs, y_load)
    x = range(1,length(tobs))
    Prev_diff = (Prev_mean[1,:] - y_load[!, "y1"]).^2
    Prev_mse = mean(Prev_diff./y_load[!,"y1"])
    Infc_diff = (Infc_mean[1,:]-y_load[!,"y2"]).^2
    Infc_mse = mean(Infc_diff./y_load[!,"y2"])

    plt1 = Plots.plot()
    Plots.plot!(plt1, x, Prev_mean[1,:], color=:black, label="Simulation mean")
    Plots.plot!(plt1, x, Prev_mean[1,:]-1.96*Prev_std[1,:], fillrange=Prev_mean[1,:]+1.96*Prev_std[1,:], fillalpha=0.35, c=1, label="")
    Plots.plot!(plt1, x, y_load[!,"y1"], line=:scatter, marker_color=:black, label="Data")

    plt2 = Plots.plot()
    Plots.plot!(plt2, x, Infc_mean[1,:], color=:black, label="Simulation mean")
    Plots.plot!(plt2, x, Infc_mean[1,:]-1.96*Infc_std[1,:], fillrange=Infc_mean[1,:]+1.96*Infc_std[1,:], fillalpha=0.35, c=1, label="")
    Plots.plot!(plt2, x, y_load[!,"y2"], line=:scatter, marker_color=:black, label="Data")
    print("Relative mean squared error in... Prev: $Prev_mse      Infc: $Infc_mse")
    return plt1, plt2 
end

# compute the resimulation errors, so the ability of randomly sampled parameter vectors from the chain to produce trajectories covering the real data
function check_chains(SDE_problem, chain, tobs, y_load; burn_in::Integer=0)
    burnin = burn_in+1
    nsim = 500
    
    Prev = Matrix{Float64}(undef, nsim, length(tobs))
    Infc = Matrix{Float64}(undef, nsim, length(tobs))
    
    # Handwritten like Ensemble problem
    for i in 1:1:nsim
        idx_iter = burn_in+rand(axes(chain.value[burnin:end,:,:], 1))
        idx_chain = rand(axes(chain.value, 3))
        parameter_chain = chain[:, ["beta", "gamma"],:]
        p = parameter_chain.value[idx_iter, :, idx_chain]
        u0_new = copy(SDE_problem.u0)
        parameter = copy(SDE_problem.p)
        parameter[1] = p[1] 
        parameter[2] = p[2]
        
        SDE_problem_new = remake(SDE_problem; u0=u0_new, p=parameter)
        # solver settings
        solve_alg = PosEM()
        solve_kwargs = (dt=1e-3, force_dtmin=true)
        nothing 
    
        sol = solve(SDE_problem_new, solve_alg; solve_kwargs...)
        sim_data = [rand(fobs(sol(t), nothing, t)) for t in tobs]

        prevalence = []
        infections = []
        for j in range(1, length(tobs))
            push!(prevalence, sim_data[j][1])
            push!(infections, sim_data[j][2])
        end
        Prev[i,:] = prevalence
        Infc[i,:] = infections
    end
    return mean(Prev; dims=1), mean(Infc; dims=1), std(Prev;dims=1), std(Infc;dims=1)    
end

# plot the resimulation error results from the above function
function data_check_sir(SDE_problem, chain, tobs, y_load; burn_in::Integer=0)
    Prev_mean, Infc_mean, Prev_std, Infc_std = check_chains(SDE_problem, chain, tobs, y_load; burn_in=burn_in)
    x = range(1,length(tobs))
    
    plt1 = Plots.plot(title = "Infectioncount")
    Plots.plot!(plt1, x, Prev_mean[1,:], label="simulation mean")
    Plots.plot!(plt1, x, Prev_mean[1,:]-1.96*Prev_std[1,:], fillrange=Prev_mean[1,:]+1.96*Prev_std[1,:], fillalpha=0.35, c=1, label="")
    Plots.plot!(plt1, x, y_load[!,"infection count"], line=:scatter, label="data")
    
    plt2 = Plots.plot(title="Antibody measurements")
    Plots.plot!(plt2, x, Infc_mean[1,:], label="simulation mean")
    Plots.plot!(plt2, x, Infc_mean[1,:]-1.96*Infc_std[1,:], fillrange=Infc_mean[1,:]+1.96*Infc_std[1,:], fillalpha=0.35, c=1, label="")
    Plots.plot!(plt2, x, y_load[!,"seroprev"], line=:scatter, label="data")

    return plt1, plt2
end

# evaluate all information about a chain at once
function evaluate_chain_sir(SDE_problem, chain, tobs, y_load, parameter_names)
    burn_in = burn_in_from_geweke(chain)
    if burn_in==size(chain)[1]-1
        print("Geweke test indicates no convergence of the chain")
        burn_in = Int(round(size(chain)[1]/2))
    end
    print("Burn In: ", burn_in)
    print("\n Abs. err. between post. mean and true values:", SDE_problem.p-mean(chain[burn_in:end]).nt.mean)
    msjd = DataFrame(MSJD=mean_squared_jumping_distance(chain, parameter_names;burn_in=burn_in).total_msjd)
    print(hcat(msjd,acceptance_rate(chain; burn_in=burn_in)))
    print("\n", DataFrame(MCMCChains.summarize(chain[burn_in:end])))
    display(autocorplot(chain[burn_in:end]))
    display(StatsPlots.plot(chain[burn_in:end]))
    plt1, plt2 = data_check_sir(SDE_problem, chain, tobs, y_load; burn_in=burn_in)
    display(plot(plt1, plt2, layout=(1,2), size=(800,350), legend=false))
    return nothing
end


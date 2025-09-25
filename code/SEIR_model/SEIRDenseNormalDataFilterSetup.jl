

#-----------------------------------------------------------------------------------------------------------------
function fobs_normal(x, p, t; scaling=2.0)
    N_infc = round(x[5]+x[6]+x[10])
    N_prev = round(x[3]+x[4]+x[6]+x[8]+x[9])
    N_total = round(x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10])
    if t in tobs
    	idx = findfirst(t .== tobs)
        return SIndependent(
            truncated(Normal(scaling*N_infc/N_total, noise_infc[idx]; check_args=false), 0.0, nothing),
            truncated(Normal(N_prev/N_total, noise_prev[idx]; check_args=false), 0.0, nothing)
        )
    end
    return SIndependent(
        truncated(Normal(scaling*N_infc/N_total, 0.1; check_args=false), 0.0, nothing),
        truncated(Normal(N_prev/N_total, 0.1; check_args=false), 0.0, nothing),
    )
end

# callback function for variant introduction
function make_event_cb(tevent::Float64, jumpsize::Float64)
    return DiscreteCallback((u,t,integrator) -> t == tevent, integrator -> integrator.u[10] += jumpsize)
end
make_event_cb(tevent, jumpsize) = make_event_cb(convert(Float64, tevent), convert(Float64, jumpsize))
tevent = true_par[4]
jumpsize = init_var
cb = make_event_cb(tevent, jumpsize)

# solver settings
solve_alg = PosEM()
solve_kwargs = (dt=1e-2, callback=cb, tstops=[tevent])
nothing

# creating the likelihood function
function likelihood(nparticles)    
    llh = let nparticles = nparticles
        p -> begin
            initial_state = [0, init_N-p[end], 0, 0, p[end], 0, 0, 0, 0, 0]
            ssm = SDEStateSpaceModel(SDE_problem, initial_state, (fobs_normal, (scaling=3.0,)), nobs, tobs, solve_alg; solve_kwargs...)
            llh_ssm = LogLikelihood_NoGradient(ssm, real_data; nparticles=nparticles)
            tevent = Float64(p[4])
            @assert tevent > 0
            @assert !isnan(tevent)
            jumpsize = Float64(init_var)
            @assert jumpsize > 0
            @assert !isnan(jumpsize)
            ssm.solve_kwargs = (dt = 1e-1, callback = make_event_cb(tevent, jumpsize), tstops = [tevent]) # NB must be of the same type (order included) as the template passed to SDEStateSpaceModel  # dt=1e-2
            scaling = p[5]
            ssm.fobs_kwargs = (; scaling)
            all(≥(0) ,p[end-1]) || return -Inf64
            p_sde = copy(ssm.sprob.p)
            p_sde[1:4] = p[1:4]
            
            @assert !any(isnan.(p_sde)) 
            return_val = llh_ssm(p_sde)
            # @assert !isnan(return_val)
            if isnan(return_val)
                return -Inf
            end
            return return_val
        end
    end
    return llh
end

# creating the posterior function
function log_posterior(nparticles)
    llh = likelihood(nparticles)
    prior = SEIR_Prior()
    post = x -> llh(x) + logpdf(prior, x)
    return post
end
#-----------------------------------------------------------------------------------------------------------------

# real observation function from petab-files
nobs = 2

function prev_comb(x, p, t)
    nom = (x[3]+x[4]+x[6]+x[8]+x[9])
    nsum = x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10]
    return nom/nsum
end

function infc_rel_nat(x, p, t; scaling::Real=2.33)
    nom = x[5]+x[6]+x[10]
    nsum = x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10]
    return scaling*nom/nsum
end

function fobs(x, p, t; scaling::Real=2.33)
    
    if t == 242
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.05, check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.05086166751331457; check_args=false), 0.0, nothing),
        )
    end
    if t == 262
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.019903244690974285; check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.05, check_args=false), 0.0, nothing),
        )
    end
    if t == 273
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.05, check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.049501502894785186; check_args=false), 0.0, nothing),
        )
    end
    if t == 304
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.027361836386980545; check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.048944430985639616; check_args=false), 0.0, nothing),
        )
    end
    if t == 308
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.05399166006542568; check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.05, check_args=false), 0.0, nothing),
        )
    end
    if t == 328
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.09578163507252513; check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.05, check_args=false), 0.0, nothing),
        )
    end
    if t == 333
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.05, check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.06107833946305408; check_args=false), 0.0, nothing),
        )
    end
    if t == 343
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.03940999395589195; check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.05, check_args=false), 0.0, nothing),
        )
    end
    if t == 353
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.0452031209168885; check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.05, check_args=false), 0.0, nothing),
        )
    end
    if t == 363
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.05, check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.0695389424489959; check_args=false), 0.0, nothing),
        )
    end
    if t == 393
        return SIndependent(
            truncated(Normal(prev_comb(x, p, t), 0.05, check_args=false), 0.0, nothing),
            truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.07212313609837218; check_args=false), 0.0, nothing),
        )
    end
    return SIndependent(
        truncated(Normal(prev_comb(x, p, t), 0.05, check_args=false), 0.0, nothing),
        truncated(Normal(infc_rel_nat(x, p, t, scaling=scaling), 0.05, check_args=false), 0.0, nothing),
    )
end

# initial values
initial_state = SDE_problem.u0;

true_params = SDE_problem.p[1:4]
true_par = 1


# callbacks for injection of variant
tevent = Float64(170.0)
jumpsize = Float64(init_var)

make_event_cb(tevent, jumpsize) = DiscreteCallback((u,t,integrator) -> t == tevent, integrator -> integrator.u[10] += jumpsize)
cb = make_event_cb(tevent, jumpsize)

# solver settings
solve_alg = PosEM()
solve_kwargs = (dt=1e-2, callback=cb, tstops=[tevent])
nothing

# creating the SDEStateSpaceModel
ssm = SDEStateSpaceModel(SDE_problem, initial_state, (fobs, (scaling=2.33,)), nobs, tobs, solve_alg; solve_kwargs...);

# creating the likelihood function

function likelihood(nparticles)
    llh = let nparticles = nparticles
        p -> begin
            initial_state = [0, init_N-p[end], 0, 0, p[end], 0, 0, 0, 0, 0]
            ssm = SDEStateSpaceModel(SDE_problem, initial_state, (fobs, (scaling=3.0,)), nobs, tobs, solve_alg; solve_kwargs...)
            llh_ssm = LogLikelihood_NoGradient(ssm, real_data; nparticles=nparticles)
            tevent = Float64(p[4])
            jumpsize = Float64(init_var)
            ssm.solve_kwargs = (dt = 0.01, callback = make_event_cb(tevent, jumpsize), tstops = [tevent]) # NB must be of the same type (order included) as the template passed to SDEStateSpaceModel
            ssm.fobs_kwargs = (scaling=Float64(p[end]),)
            scaling = p[5]
            ssm.fobs_kwargs = (; scaling)
            all(≥(0) ,p[begin:end]) || return -Inf64 # negative rates are not possible
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
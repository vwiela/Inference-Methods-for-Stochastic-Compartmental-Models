# observation model
nobs = 1

function infec_counts(x, p, t) #infection counts
    return x[2]
end

function fobs(x, p, t)
    if t in tobs
    	idx = findfirst(t .== tobs)
        return SIndependent(
            truncated(Normal(infec_counts(x,p,t), noise_infc[idx]; check_args=false), 0.0, nothing)
        )
    end
    return SIndependent(
            truncated(Normal(infec_counts(x,p,t), 0.2; check_args=false), 0.0, nothing)
        )
end;

# initial values
initial_state = problem.u0;

true_params = problem.p

# solver settings
solve_alg = PositiveEM()
solve_kwargs = (dt=1e-2, dense=true, force_dtmin=true)
nothing

# creating the SDEStateSpaceModel
ssm = SDEStateSpaceModel(problem, initial_state, fobs, nobs, tobs, solve_alg; solve_kwargs...);

# define likelihood function
function log_likelihood(nparticles)
    llh_ssm = LogLikelihood_NoGradient(ssm, real_data;nparticles=nparticles)
    llh = let llh_ssm =llh_ssm, ssm=ssm
        p -> llh_ssm(p)
    end
    return llh
end

function log_posterior(nparticles)
    llh = log_likelihood(nparticles)
    prior = SIS_Prior()
    post = x -> llh(x) + logpdf(prior, x)
    return post
end

# define prior for parameters for sampling startpoints and in adding prior value to sampling.
struct SIS_Prior end
function Random.rand(rng::AbstractRNG, d::SIS_Prior) 
    β = rand(Uniform(0.2, 1.0))
    γ_inv = rand(Uniform(1.0, 20.0))
    while true
        if β*γ_inv >= 1.15 && β*γ_inv <= 5.0
            break
        end
        # β = rand(Uniform(0.0, 1.0))
        γ_inv = rand(Uniform(1.0, 20.0))
    end
    return [β, 1/γ_inv]
end
Random.rand(d::SIS_Prior) = rand(Random.default_rng(), d)


function Distributions.logpdf(::SIS_Prior, x)
    if x[1]/x[2] >= 1.1 && x[1]/x[2] <= 2.5
        return logpdf(Uniform(0.2, 1.0), x[1]) + logpdf(Uniform(1.0, 20.0), 1/x[2])
    else
        return -Inf
    end
end
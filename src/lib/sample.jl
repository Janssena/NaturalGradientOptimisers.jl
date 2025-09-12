"""
    sample_z(ps, st; dist=:gaussian)

Takes a random sample from the distributions defined in the parameters `ps` based
on the corresponding epsilon samples in the state `st`. The `dist` keyword 
indicates the type of distribution to look for.
"""
function sample_z(ps, st; dist::Symbol=:gaussian) 
    dist_ = Static.static(dist)
    return fmap_with_path(ps; exclude = _get_dist_walk(dist_)) do kp, x
        (z = sample_dist(dist_, x, getkeypath(st, kp)), )
        # TODO: this would be the place to merge(x, (z = sample_dist, ))
        # It would be nice to remove the distribution parameters here.
        # Alternatively, it also makes sense to force users to isolate 
        # distribution parameters from other parameters.
    end
end

##### Gaussian distributions

_get_dist_walk(::Static.StaticSymbol{:gaussian}) = _walk_until_gaussian

_walk_until_gaussian(::KeyPath, x) = _walk_until_gaussian(x)

_get_dist_parameters(::Static.StaticSymbol{:gaussian}) = 
    (:μ, [:σ, :σ², :Σ, :L, :S, :T])

function _walk_until_gaussian(x)
    if applicable(keys, x)
        mu_symbol, var_symbols = _get_dist_parameters(Static.static(:gaussian))
        return mu_symbol in keys(x) && !isempty(filter(∈(var_symbols), keys(x)))
    else
        return false # If x does not have a keys function, don't walk through it
    end    
end

sample_dist(::Static.StaticSymbol{:gaussian}, ps, st) = sample_gauss(ps, st)

# These are the function users should extend if their parameters 
# (i.e. distributions) contain additional variables:
# mean, precision parameterization:
sample_gauss(ps::NamedTuple{(:μ,:S)}, st) = ps.μ + cholesky(ps.S).L \ st.epsilon
sample_gauss(ps::NamedTuple{(:μ,:T)}, st) = ps.μ + ps.T \ st.epsilon
# mean, covariance parameterization:
sample_gauss(ps::NamedTuple{(:μ,:Σ)}, st) = ps.μ + cholesky(ps.Σ).L * st.epsilon
sample_gauss(ps::NamedTuple{(:μ,:L)}, st) = ps.μ + ps.L * st.epsilon
sample_gauss(ps::NamedTuple{(:μ,:σ²)}, st) = ps.μ + sqrt.(ps.σ²) .* st.epsilon
sample_gauss(ps::NamedTuple{(:μ,:σ)}, st) = ps.μ + σ .* st.epsilon
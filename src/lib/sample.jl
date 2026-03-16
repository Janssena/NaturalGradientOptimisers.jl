"""
    sample_z(ps, st; dist=:gaussian, exclude = nothing)

Takes a random sample from the distributions defined in the parameters `ps` based
on the corresponding epsilon samples in the state `st`. The `dist` keyword 
indicates the type of distribution to look for. One can set the `exclude` keyword 
to exclude certain paths from sampling based on (one of) their parent key(s) in `ps` 
and `st`. 
"""
function sample_z(ps, st; dist::Symbol=:gaussian, exclude = nothing) 
    dist_ = Static.static(dist)
    return fmap_with_path(ps; exclude = _get_dist_walk(dist_)) do kp, x
        if !isnothing(exclude) && exclude in kp
            return x
        else
            (z = sample_dist(dist_, x, getkeypath(st, kp)), )
        end
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

sample_dist(::Static.StaticSymbol{:gaussian}, ps::NamedTuple{<:Any, <:Tuple{<:AbstractVector{<:Real}, Vararg}}, st) = 
    sample_gauss(ps, st)

# Special version that works with parameters that feature vectors of 
# distributions. Assumes that the first two elements correspond to the gaussian 
# parameters.
sample_dist(::Static.StaticSymbol{:gaussian}, ps::NamedTuple{<:Any, <:Tuple{<:AbstractVector{<:AbstractArray}, Vararg}}, st) = 
    map(eachindex(ps.μ)) do i
        sample_gauss(_take_idx(ps, i), _take_idx(st, i))
    end # TODO: This might be a lot of unnecessary allocations where we could instead broadcast
        # We would have to create broadcasted versions of the the below options though

# These are the function users should extend if their parameters 
# (i.e. distributions) contain additional variables:
# mean, precision parameterization:
sample_gauss(ps::NamedTuple{(:z,)}, st) = ps.z
sample_gauss(ps::NamedTuple{(:μ,:S)}, st) = ps.μ + cholesky(ps.S).L \ st.epsilon
sample_gauss(ps::NamedTuple{(:μ,:T)}, st) = ps.μ + ps.T \ st.epsilon
# mean, covariance parameterization:
sample_gauss(ps::NamedTuple{(:μ,:Σ)}, st) = ps.μ + cholesky(ps.Σ).L * st.epsilon
sample_gauss(ps::NamedTuple{(:μ,:L)}, st) = ps.μ + ps.L * st.epsilon
sample_gauss(ps::NamedTuple{(:μ,:σ²)}, st) = ps.μ + sqrt.(ps.σ²) .* st.epsilon
# This assumes that σ is already constrained
sample_gauss(ps::NamedTuple{(:μ,:σ)}, st) = ps.μ + softplus.(σ) .* st.epsilon
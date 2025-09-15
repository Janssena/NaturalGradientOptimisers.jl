module NaturalGradientOptimisers
    
import LinearAlgebra: Symmetric, Diagonal, cholesky, I
import Optimisers
import Random
import Static

using Functors

_kp_has_key(kp::KeyPath, key::Symbol) = 
        kp[end] == key || (kp[end-1] == key && kp[end] isa Integer)

_take_idx(x, i) = x
_take_idx(x::AbstractVector{<:AbstractArray}, i) = x[i]
function _take_idx(x::NamedTuple, i) 
    keys_ = keys(x)
    values_ = map(keys_) do key # Almost a fmap, but does not recurse into indexes
        _take_idx(x[key], i)
    end
    return NamedTuple{keys_}(values_)
end

include("lib/descent.jl");
include("lib/interface.jl");

export  NaturalDescent, NaturalDescentRule, update_state!, 
        update_epsilon!

include("lib/gradients.jl");
include("lib/sample.jl");

export  estimate_covariance_gradient_from_dz, dlogq!, sample_z, 
        sample_gauss

end # module NaturalGradientOptimisers

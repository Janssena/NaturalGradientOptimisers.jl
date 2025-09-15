module NaturalGradientOptimisers
    
import LinearAlgebra: Symmetric, Diagonal, cholesky, I
import Optimisers
import Random
import Static

using Functors

_kp_has_key(kp::KeyPath, key::Symbol) = 
        kp[end] == key || (kp[end-1] == key && kp[end] isa Integer)

include("lib/descent.jl");
include("lib/interface.jl");

export  NaturalDescent, NaturalDescentRule, update_state!, 
        update_epsilon!

include("lib/gradients.jl");
include("lib/sample.jl");

export  estimate_covariance_gradient_from_dz, dlogq!, sample_z, 
        sample_gauss

end # module NaturalGradientOptimisers

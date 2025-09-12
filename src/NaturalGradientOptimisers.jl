module NaturalGradientOptimisers
    
import LinearAlgebra: Symmetric, Diagonal, cholesky, I
import Optimisers
import Static

using Functors

include("lib/descent.jl");
include("lib/interface.jl");

export NaturalDescent, NaturalDescentRule, update_state!

include("lib/gradients.jl");
include("lib/sample.jl");

export estimate_covariance_gradient_from_dz, dlogq!, sample_z, sample_gauss

end # module NaturalGradientOptimisers

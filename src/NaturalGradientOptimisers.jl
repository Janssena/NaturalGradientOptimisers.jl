module NaturalGradientOptimisers
    
import LinearAlgebra: Symmetric, Diagonal, I
import Optimisers
import Static

using Functors

include("lib/descent.jl");
include("lib/interface.jl");

export NaturalDescent, NaturalDescentRule, update_state!

end # module NaturalGradientOptimisers

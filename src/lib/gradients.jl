_walk_until_z(::KeyPath, x) = _walk_until_z(x)
_walk_until_z(x) = applicable(keys, x) ? :z in keys(x) : false

function estimate_covariance_gradient_from_dz(grad::NamedTuple, ps, st::NamedTuple)
    # TODO: this will not work correctly if we have z amongst the gradients of 
    # the first level. The z gradients are expected to be a subset of model 
    # parameters, i.e. never (z = ..., other = (z = ..., )). The `other` in the
    # current solution will never be traversed.
    if _walk_until_z(grad) # i.e. grad contains z in the first level.
        ∇Σ = estimate_covariance_gradient(grad.z, ps, st.epsilon)
        return NamedTuple{keys(ps)}((grad.z, ∇Σ))
    else # walk through the gradients to find ∇z
        return fmap_with_path(grad; exclude = _walk_until_z) do kp, ∇
            q = getkeypath(ps, kp)
            stᵢ = getkeypath(st, kp)
            ∇Σ = estimate_covariance_gradient(∇.z, q, stᵢ.epsilon)
            return NamedTuple{keys(q)}((∇.z, ∇Σ))
        end
    end
end

function estimate_covariance_gradient(dz::AbstractVector, ps::NamedTuple{(:μ,:S)}, ϵ::AbstractVector)
    S̄ = (cholesky(ps.S).L' * ϵ) .* dz' # Σ⁻¹ (z - μ) == chol(S)' * ϵ
    return 0.25 .* (S̄ + S̄') # Ensure symmetry
end

function estimate_covariance_gradient(dz::AbstractVector, ps::NamedTuple{(:μ,:Σ)}, ϵ::AbstractVector)
    S̄ = (cholesky(ps.Σ).L' \ ϵ) .* dz' # Σ⁻¹ (z - μ) == chol(Σ)' \ ϵ
    return 0.25 .* (S̄ + S̄') # Ensure symmetry
end

function estimate_covariance_gradient(dz::AbstractVector, ps::NamedTuple{(:μ,:σ²)}, ϵ::AbstractVector)
    s̄ = (ϵ ./ sqrt.(ps.σ²)) .* dz # σ⁻² (z - μ) == ϵ ./ σ
    return 0.5 .* s̄
end

"""
    dlogq!(grad, ps)

Adds the gradients for μ and Σ (or σ², S, etc...) with respect to 
𝔼_q[∇z log q(z)] to the passed `grad` in place.

Note: this function only adds the part of the gradient that does not have 
expectation zero. This thus simply performs ∇μ_full = ∇μ and 
∇Σ_full = ∇Σ - 0.5 ⋅ Σ⁻¹.
"""
function dlogq!(grad, ps) 
    if _walk_until_gaussian(grad) # i.e. grad contains the gaussian parameters in the first level
        # TODO: this will not work correctly if we have the distribution 
        # parameters amongst the gradients of the first level. The gradients are 
        # expected to be a subset of model parameters, i.e. never 
        # (μ = ..., Σ = ..., other = (μ = ..., Σ = ..., )). In the current setup,
        # we will never traverse the elements of `other`.
        subtract_logq_grad!(grad, ps) # This will throw an error if there are more elements
        # TODO: if length(keys(grad)) > 1 we could address this by running 
        # dlogq!(grad[Not(:z)], ps[:Not(:z)]), but we should instead look for a 
        # different solution that also works for estimate_covariance_gradient_from_dz
    else # walk through the gradients to find ∇z
        fmap_with_path(grad; exclude = _walk_until_gaussian) do kp, ∇
            subtract_logq_grad!(∇, getkeypath(ps, kp))
            return ∇ # We are not using the return essentially
        end
    end
    
    return nothing
end

function subtract_logq_grad!(∇::NamedTuple{(:μ,:σ²)}, ps::NamedTuple{(:μ,:σ²)})
    ∇.σ² .= ∇.σ² - eltype(ps.σ²)(0.5) * (one(T) ./ ps.σ²)
    return nothing
end

function subtract_logq_grad!(∇::NamedTuple{(:μ,:Σ)}, ps::NamedTuple{(:μ,:Σ)})
    ∇.Σ .= ∇.Σ - eltype(ps.Σ)(0.5) * inv(ps.Σ)
    return nothing
end

function subtract_logq_grad!(∇::NamedTuple{(:μ,:S)}, ps::NamedTuple{(:μ,:S)})
    ∇.S .= ∇.S - eltype(ps.S)(0.5) * ps.S
    return nothing
end
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

estimate_covariance_gradient(dz::AbstractVector{<:AbstractArray}, ps, ϵ::AbstractVector{<:AbstractArray}) = 
    map(eachindex(dz)) do i
        estimate_covariance_gradient(dz[i], _take_idx(ps, i), ϵ[i])
    end

function estimate_covariance_gradient(dz::AbstractVector{<:Real}, ps::NamedTuple{(:μ,:S)}, ϵ::AbstractVector{<:Real})
    S̄ = (cholesky(ps.S).L' * ϵ) .* dz' # Σ⁻¹ (z - μ) == chol(S)' * ϵ
    return 0.25 .* (S̄ + S̄') # Ensure symmetry
end

function estimate_covariance_gradient(dz::AbstractVector{<:Real}, ps::NamedTuple{(:μ,:Σ)}, ϵ::AbstractVector{<:Real})
    S̄ = (cholesky(ps.Σ).L' \ ϵ) .* dz' # Σ⁻¹ (z - μ) == chol(Σ)' \ ϵ
    return 0.25 .* (S̄ + S̄') # Ensure symmetry
end

function estimate_covariance_gradient(dz::AbstractVector{<:Real}, ps::NamedTuple{(:μ,:σ²)}, ϵ::AbstractVector{<:Real})
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
    if ps.σ² isa AbstractVector{<:AbstractArray}
        T = eltype(first(ps.σ²))
        for i in eachindex(ps.σ²)
            ∇.σ²[i] .= ∇.σ²[i] - T(0.5) .* (one(T) ./ ps.σ²[i])
        end
    else
        T = eltype(ps.σ²)
        ∇.σ² .= ∇.σ² - T(0.5) .* (one(T) ./ ps.σ²)
    end
    
    return nothing
end

function subtract_logq_grad!(∇::NamedTuple{(:μ,:Σ)}, ps::NamedTuple{(:μ,:Σ)})
    if ps.Σ isa AbstractVector{<:AbstractArray}
        T = eltype(first(ps.Σ))
        for i in eachindex(ps.Σ)
            ∇.Σ[i] .= ∇.Σ[i] - 0.5 * inv(ps.Σ[i])    
        end
    else
        T = eltype(ps.Σ)
        ∇.Σ .= ∇.Σ - 0.5 * inv(ps.Σ)
    end
    return nothing
end

function subtract_logq_grad!(∇::NamedTuple{(:μ,:S)}, ps::NamedTuple{(:μ,:S)})
    if ps.S isa AbstractVector{<:AbstractArray}
        T = eltype(first(ps.S))
        for i in eachindex(ps.S)
            ∇.S[i] .= ∇.S[i] - 0.5 * ps.S[i]
        end
    else
        T = eltype(ps.S)
        ∇.S .= ∇.S - 0.5 * ps.S
    end

    return nothing
end
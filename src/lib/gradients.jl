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
            # TODO: the below should go into a function and should also be used in the above condition.
            names = keys(q)
            values = map(names) do name
                if name == :μ
                    return ∇.z
                elseif name == :Σ || name == :σ² # How do we handle other names?
                    return ∇Σ
                else
                    return nothing
                end
            end
            return NamedTuple{names}(values)
        end
    end
end

estimate_covariance_gradient(dz::AbstractVector{<:AbstractArray}, ps, ϵ::AbstractVector{<:AbstractArray}) = 
    map(eachindex(dz)) do i
        estimate_covariance_gradient(dz[i], _take_idx(ps, i), ϵ[i])
    end

function estimate_covariance_gradient(dz::AbstractVector{<:Real}, ps::NamedTuple{(:μ,:S)}, ϵ::AbstractVector{T}) where T<:Real
    S̄ = (cholesky(ps.S).L' * ϵ) .* dz' # Σ⁻¹ (z - μ) == chol(S)' * ϵ
    return Symmetric(T(0.25) * (S̄ + S̄')) # Ensure symmetry
end

function estimate_covariance_gradient(dz::AbstractVector{<:Real}, ps::NamedTuple{(:μ,:Σ)}, ϵ::AbstractVector{T}) where T<:Real
    S̄ = (cholesky(ps.Σ).L' \ ϵ) .* dz' # Σ⁻¹ (z - μ) == chol(Σ)' \ ϵ
    return Symmetric(T(0.25) * (S̄ + S̄')) # Ensure symmetry
end

function estimate_covariance_gradient(dz::AbstractVector{<:Real}, ps::NamedTuple{(:μ,:σ²)}, ϵ::AbstractVector{T}) where T<:Real
    s̄ = (ϵ ./ sqrt.(ps.σ²)) .* dz # σ⁻² (z - μ) == ϵ ./ σ
    return T(0.5) * s̄
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
        add_logq_grad!(grad, ps) # This will throw an error if there are more elements
        # TODO: if length(keys(grad)) > 1 we could address this by running 
        # dlogq!(grad[Not(:z)], ps[:Not(:z)]), but we should instead look for a 
        # different solution that also works for estimate_covariance_gradient_from_dz
    else # walk through the gradients to find ∇z
        fmap_with_path(grad; exclude = _walk_until_gaussian) do kp, ∇
            add_logq_grad!(∇, getkeypath(ps, kp))
            return ∇ # We are not using the return essentially
        end
    end
    
    return nothing
end

function add_logq_grad!(∇::NamedTuple{(:μ,:σ²)}, ps::NamedTuple{(:μ,:σ²)})
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

function add_logq_grad!(∇::NamedTuple{(:μ,:Σ)}, ps::NamedTuple{(:μ,:Σ)})
    if ps.μ isa AbstractVector{<:Real}
        T = eltype(∇.Σ)
        ∇.Σ .= ∇.Σ - T(0.5) * inv(ps.Σ)
    else
        map(∇.Σ, ps.Σ) do ∇Σ, psΣ
            if ∇Σ isa AbstractArray{<:Real}
                T = eltype(∇Σ)
                ∇Σ .= ∇Σ - T(0.5) * inv(psΣ)
            elseif ∇Σ isa AbstractVector{<:AbstractArray{<:Real}}
                for i in eachindex(∇Σ)
                    T = eltype(∇Σ[i])
                    ∇Σ[i] .= ∇Σ[i] - T(0.5) * inv(psΣ[i])    
                end
            else
                throw(ErrorException("Not implemented!"))
            end
        end
    end
    # if ps.Σ isa AbstractVector{<:AbstractArray}
    #     T = eltype(first(ps.μ))
    #     for i in eachindex(ps.Σ)
    #         ∇.Σ[i] .= ∇.Σ[i] - T(0.5) * inv(ps.Σ[i])    
    #     end
    # else
    #     T = eltype(ps.μ)
    #     ∇.Σ .= ∇.Σ - T(0.5) * inv(ps.Σ)
    # end
    return nothing
end

# TODO: Correct based on the above example
function add_logq_grad!(∇::NamedTuple{(:μ,:S)}, ps::NamedTuple{(:μ,:S)})
    if ps.S isa AbstractVector{<:AbstractArray}
        T = eltype(first(ps.μ))
        for i in eachindex(ps.S)
            dS = -T(0.5) * ps.S[i]
            ∇.S[i] .= ∇.S[i] + dS
        end
    else
        T = eltype(ps.μ)
        dS = -T(0.5) * ps.S
        ∇.S .= ∇.S + dS
    end

    return nothing
end

function dlogp!(grad, ps, priors)
    if _walk_until_gaussian(grad) # i.e. grad contains the gaussian parameters in the first level
        # TODO: this will not work correctly if we have the distribution 
        # parameters amongst the gradients of the first level. The gradients are 
        # expected to be a subset of model parameters, i.e. never 
        # (μ = ..., Σ = ..., other = (μ = ..., Σ = ..., )). In the current setup,
        # we will never traverse the elements of `other`.
        add_logp_grad!(grad, ps, priors) # This will throw an error if there are more elements
        # TODO: if length(keys(grad)) > 1 we could address this by running 
        # dlogq!(grad[Not(:z)], ps[:Not(:z)]), but we should instead look for a 
        # different solution that also works for estimate_covariance_gradient_from_dz
    else # walk through the gradients to find ∇z
        fmap_with_path(grad; exclude = _walk_until_gaussian) do kp, ∇
            add_logp_grad!(∇, getkeypath(ps, kp), getkeypath(priors, kp))
            return ∇ # We are not using the return essentially
        end
    end
    
    return nothing
end

# TODO: Make use of gradlogpdf instead of this manual implementation
function subtract_logp_grad!(∇::NamedTuple{(:μ,:Σ)}, q, p)
    S = inv(p.Σ)
    T = eltype(S)
    du = -S * (q.μ - p.μ)
    ∇.μ .= ∇.μ - du
    dΣ = -T(0.5) * S
    ∇.Σ .= ∇.Σ + dΣ

    return nothing
end

function subtract_logp_grad!(∇::NamedTuple{(:z, )}, q, p)
    S = inv(p.Σ)
    du = -S * (q.μ - p.μ)
    ∇.z .= ∇.z - du

    return nothing
end






function dgauss_kl!(grad, ps, priors) 
    if _walk_until_gaussian(grad) # i.e. grad contains the gaussian parameters in the first level
        # TODO: this will not work correctly if we have the distribution 
        # parameters amongst the gradients of the first level. The gradients are 
        # expected to be a subset of model parameters, i.e. never 
        # (μ = ..., Σ = ..., other = (μ = ..., Σ = ..., )). In the current setup,
        # we will never traverse the elements of `other`.
        add_gauss_kl_grad!(grad, ps, priors) # This will throw an error if there are more elements
        # TODO: if length(keys(grad)) > 1 we could address this by running 
        # dlogq!(grad[Not(:z)], ps[:Not(:z)]), but we should instead look for a 
        # different solution that also works for estimate_covariance_gradient_from_dz
    else # walk through the gradients to find ∇z
        fmap_with_path(grad; exclude = _walk_until_gaussian) do kp, ∇
            add_gauss_kl_grad!(∇, getkeypath(ps, kp), getkeypath(priors, kp))
            return ∇ # We are not using the return essentially
        end
    end
    
    return nothing
end

function gauss_kl_grad(q::NamedTuple{(:μ,:Σ)}, p)
    S₀ = inv(p.Σ)
    T = eltype(S₀)
    return (μ = S₀ * (q.μ - p.μ), Σ = T(0.5) * (S₀ - inv(q.Σ)))
end

function add_gauss_kl_grad!(∇, q, p)
    ∇kl = gauss_kl_grad(q, p)
    ∇.μ .= ∇.μ + ∇kl.μ
    var_grad = ∇.Σ + ∇kl.Σ
    ∇.Σ .= ∇.Σ isa Symmetric ? Symmetric(var_grad) : var_grad
    
    return nothing
end
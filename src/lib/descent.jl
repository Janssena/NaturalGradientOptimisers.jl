abstract type NaturalDescentRule <: Optimisers.AbstractRule end

#####

# TODO:
# struct ImplicitNaturalDescent{MF<:Static.StaticBool} <: Optimisers.AbstractRule
#     eta # Learning rate
#     delta # Damping parameter with decay (latter is a TODO)
#     ImplicitNaturalDescent(eta = 0.1f0, delta = 0.f0; mean_field::Bool = true) = 
#         new{static(mean_field)}(eta, delta)
# end

# Optimisers.init(::ImplicitNaturalDescent{Static.False}, x::AbstractVector) = 
#     (copy(x), one.(x))

# Optimisers.init(::ImplicitNaturalDescent{Static.True}, x::AbstractVector) = 
#     (copy(x), Symmetric(collect(Diagonal(one.(x)))))

# function Optimisers.apply!(
#     o::ImplicitNaturalDescent{Static.False}, 
#     state, 
#     x::AbstractVector{T}, 
#     dx::AbstractVector{<:Real}) where {T<:Real}
    
#     η, δ = convert(T, o.eta), convert(T, o.delta)
#     mₜ, Sₜ = state
#     Σₜ = inv(Σₜ)
#     if δ > 0 # Damping
#         δ_ = one(T) .- (diag(Σₜ) ./ (one(T) .+ (one(T) / δ) .* diag(Σₜ)))
#         δ⁻¹I = Diagonal(fill(one(T) / δ_, length(x)))
#         F = Σₜ - Σₜ * inv(δ⁻¹I + Σₜ) * Σₜ
#     else
#         F = Σₜ
#     end

#     # Approximate hessian ∇²z L(z)
#     S̄ = Sₜ * (x - mₜ) * dx' # x ~ N(mₜ, Σₜ)
#     d²x = 0.25 * (S̄ + S̄') - 0.5 * Sₜ

#     ĝ₁ = T.(F * dx)
#     ĝ₂ = T.(-2 * d²x)

#     mₜ₊₁ = mₜ - η * ĝ₁
#     Sₜ₊₁ = Sₜ - η * ĝ₂ + (abs2(η) / 2) * ĝ₂ * Σₜ * ĝ₂

#     return (mₜ₊₁, Sₜ₊₁), η * ĝ₁
# end

##### Explicit natural descent.


struct NaturalDescent <: NaturalDescentRule
    eta # Learning rate
    delta # Damping parameter with decay
    NaturalDescent(eta = 0.1f0, delta = 0.f0) = new(eta, delta)
end

##### NaturalDescentMean

struct NaturalDescentMean <: NaturalDescentRule
    eta # Learning rate
    delta # Damping parameter
end

Optimisers.init(::NaturalDescentMean, x::AbstractVector) = 
    (collect(Diagonal(one.(x))), )
    
Optimisers.init(::NaturalDescentMean, ::AbstractVector, var::AbstractMatrix) = 
    (copy(var), )

Optimisers.init(::NaturalDescentMean, ::AbstractVector, var::AbstractVector) = 
    (collect(Diagonal(copy(var))), )
  
function Optimisers.apply!(
    o::NaturalDescentMean, 
    state, 
    x::AbstractVector{T}, 
    dx::AbstractVector{<:Real}) where {T<:Real}
    
    η, δ = convert(T, o.eta), convert(T, o.delta)
    (Σₜ, ) = state
    
    if δ > 0 # Damping
        δ_ = one(T) .- (diag(Σₜ) ./ (one(T) .+ (one(T) / δ) .* diag(Σₜ)))
        δ⁻¹I = Diagonal(fill(one(T) / δ_, length(x)))
        F = Σₜ - Σₜ * inv(δ⁻¹I + Σₜ) * Σₜ
    else
        F = Σₜ
    end
    ĝ = T.(F * dx)

    return state, η * ĝ
end

##### NaturalDescentVariance

struct NaturalDescentVariance <: NaturalDescentRule
    eta
end

Optimisers.init(::NaturalDescentVariance, x::AbstractArray) = tuple()

function Optimisers.apply!(
    o::NaturalDescentVariance, 
    state, 
    Σ::AbstractMatrix{T}, 
    dΣ::AbstractMatrix) where {T<:Real}
    
    η = convert(T, o.eta)
    ĝ = 2 * (Σ * (dΣ) * Σ)
    return state, η * ĝ - (abs2(η) / 2) * (ĝ * inv(Σ) * ĝ)
end

function Optimisers.apply!(
    o::NaturalDescentVariance, 
    state, 
    σ²::AbstractVector{T}, 
    dσ²::AbstractVector{<:Real}) where {T<:Real}

    η = convert(T, o.eta)
    ĝ = 2 * (σ² .* dσ² .* σ²)
    return state, (η * ĝ) - (abs2(η) / 2) * (ĝ .* (one(T) ./ σ²) .* ĝ)
end

struct NaturalDescentPrecision <: NaturalDescentRule
    eta
end

Optimisers.init(::NaturalDescentPrecision, x::AbstractArray) = tuple()

function Optimisers.apply!(
    o::NaturalDescentPrecision, 
    state, 
    S::AbstractMatrix{T}, 
    dS::AbstractMatrix) where {T<:Real}
    
    η = convert(T, o.eta)
    ĝ = -2 * dS
    # Original equation says S - t * ĝ + t² / 2 * ĝ * S⁻¹ * ĝ
    # But since we are doing x - dx later this becomes  S - t * ĝ (-)! t² / 2...
    return state, η * ĝ - (abs2(η) / 2) * (ĝ * inv(S) * ĝ)
end

function Optimisers.apply!(
    o::NaturalDescentPrecision, 
    state, 
    s::AbstractVector{T}, 
    ds::AbstractVector{<:Real}) where {T<:Real}

    η = convert(T, o.eta)
    ĝ = -2 * ds
    return state, (η * ĝ) - (abs2(η) / 2) * (ĝ .* (one(T) ./ s) .* ĝ)
end

Optimisers.subtract!(x::Symmetric, x̄) = 
    Optimisers.maywrite(x) ? Symmetric(x .= x .- x̄) : Symmetric(eltype(x).(x .- x̄))

# """
# Note that the applied gradient here is with respect to S = Σ⁻¹ so we need a 
# custom update.
# """
# function Optimisers._update!(ℓ::Optimisers.Leaf{R,S}, x; grads, params) where {R<:NaturalDescentVariance,S}
#     haskey(params, (ℓ,x)) && return params[(ℓ,x)]
#     ℓ.frozen && return x
#     params[(ℓ,x)] = if haskey(grads, ℓ)
#         ℓ.state, x̄′ = Optimisers.apply!(ℓ.rule, ℓ.state, x, grads[ℓ]...)
#         return inv_subtract(x, x̄′)
#     else
#       x # no gradient seen
#     end
# end

# inv_subtract(x::AbstractVector{T}, x̄′::AbstractVector{<:Real}) where {T<:Real} = 
#     T.(one(T) ./ ((one(T) ./ x) - x̄′))

# _inv_subtract(x::AbstractMatrix{T}, x̄′::AbstractMatrix) where {T<:Real} = 
#     T.(inv(I - x * x̄′) * x)

# Conserve Symmetric type:
# inv_subtract(x::Symmetric, x̄′::AbstractMatrix) = 
#     Symmetric(_inv_subtract(x, x̄′))

# struct NaturalDescentCholesky <: NaturalDescentRule
#     eta
# end # TODO: This needs the gradient wrt Σ, which is easiest to get when approximating (e.g. using reparameterized gradient)

# # This version calculates the gradient wrt Σ as a reparameterization of ∇zL
# struct ImplicitNaturalDescent <: NaturalDescentRule
#     eta # Learning rate
#     lambda # Damping parameter with decay
# end # It might make sense to differentiate between an implicit { μ, Σ }, { μ, C }, and { μ, T } parameterization in here
# # A benefit of { μ, T } is that we can update T before updating μ.
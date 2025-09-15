abstract type NaturalDescentRule <: Optimisers.AbstractRule end

struct NaturalDescent{T<:Static.StaticSymbol} <: NaturalDescentRule
    eta # Learning rate
    delta # Damping parameter with decay
    NaturalDescent(eta = 0.1f0, delta = 0.f0) = new(eta, delta)
end

##### NaturalDescentMean

struct NaturalDescentMean <: NaturalDescentRule
    eta # Learning rate
    delta # Damping parameter
    # TODO: Momentum
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
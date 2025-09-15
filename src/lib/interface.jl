# TODO: add extension for ComponentArrays?
Optimisers._setup(rule::NaturalDescentRule, x::NamedTuple; cache) = 
    fmapstructure_with_path(x; cache, walk = Optimisers.TrainableStructWalkWithPath()) do kp, x_
        error_string = "No μ, Σ, or σ² found in key path $(kp). Run `setup(natural_rule, fallback_rule, ps)` instead to use fallback rule."
        if _kp_has_key(kp, :μ)
            inner_rule = NaturalDescentMean(rule.eta, rule.delta)
            # get current variance:
            var_ = _get_corresponding_var(kp, x)
            if isnothing(var_)
                throw(ErrorException(error_string))
            end
            ℓ = Optimisers.Leaf(inner_rule, Optimisers.init(inner_rule, x_, var_)) 
        elseif _kp_has_key(kp, :Σ) || _kp_has_key(kp, :σ²) || _kp_has_key(kp, :S)
            type = kp[end] == :S ? NaturalDescentPrecision : NaturalDescentVariance
            inner_rule = type(rule.eta)
            ℓ = Optimisers.Leaf(inner_rule, Optimisers.init(inner_rule, x_)) 
        else
            throw(ErrorException(error_string))
        end

        return ℓ
    end

Optimisers._setup(rule::NaturalDescentRule, fallback_rule::Optimisers.AbstractRule, x::NamedTuple; cache = nothing) = 
    fmapstructure_with_path(x; cache, walk = Optimisers.TrainableStructWalkWithPath()) do kp, x_
        if _kp_has_key(kp, :μ)
            inner_rule = NaturalDescentMean(rule.eta, rule.delta)
            # get current variance:
            var_ = _get_corresponding_var(kp, x)
            if isnothing(var_)
                throw(ErrorException(error_string))
            end
            ℓ = Optimisers.Leaf(inner_rule, Optimisers.init(inner_rule, x_, var_)) 
        elseif _kp_has_key(kp, :Σ) || _kp_has_key(kp, :σ²) || _kp_has_key(kp, :S)
            type = _kp_has_key(kp, :S) ? NaturalDescentPrecision : NaturalDescentVariance
            inner_rule = type(rule.eta)
            ℓ = Optimisers.Leaf(inner_rule, Optimisers.init(inner_rule, x_)) 
        else
            ℓ = Optimisers.Leaf(fallback_rule, Optimisers.init(fallback_rule, x_)) 
        end

        return ℓ
    end

_is_var_symbol(s::Symbol) = s == :Σ || s == :σ² || s == :S

function _get_corresponding_var(kp::KeyPath, x)
    parent = getkeypath(x, kp[1:end-1])
    var_symbol = filter(_is_var_symbol, keys(parent))
    if isempty(var_symbol)
        return nothing
    end
    var_kp = replace(kp, :μ => only(var_symbol))
    value = getkeypath(x, var_kp)
    return only(var_symbol) == :S ? inv(value) : value
end

Base.replace(kp::KeyPath, x::Pair) = KeyPath(replace(kp.keys, x))

function update_state!(tree, x)
    fmap_with_path(x; walk = Optimisers.TrainableStructWalkWithPath()) do kp, x_
        if _kp_has_key(kp, :μ)
            var_ = _get_corresponding_var(kp, x)
            var_mat = var_ isa AbstractVector ? collect(Diagonal(var_)) : Symmetric(var_)
            o = getkeypath(tree, kp)
            o.state[1] .= copy(var_mat)
        end
        return x_
    end

    return nothing
end

function update_epsilon!(rng::Random.AbstractRNG, st::NamedTuple)
    fmap_with_path(st) do kp, x
        if _kp_has_key(kp, :epsilon)
            x .= randn(rng, eltype(x), size(x))
        end
        return x
    end
    return nothing
end
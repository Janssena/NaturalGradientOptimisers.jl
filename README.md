# NaturalGradientOptimisers.jl

This is a julia package extending Optimisers.jl to add optimization methods based on natural gradient descent. 

### Natural Gradient Descent

Natural gradient descent is an optimization method that accounts for the geometry of the parameter space. Instead of updating parameters with respect to the raw Euclidean gradient, it rescales updates using the Fisher information matrix, which captures the local curvature of the parameter space.

This adjustment often leads to:

* Faster convergence compared to standard gradient descent.
* More stable training, especially in probabilistic models or Variational Inference.
* Better use of curvature information without the heavy cost of full second-order methods.

### How to use the package.

This package adopts the Lux.jl framework of separating learnable and fixed parameters into a parameter object `ps` and a state object `st`. The keys in these two objects match. We search `ps` for the parameters of a distribution and adds parameter specific optimizers (NaturalDescentMean or NaturalDescentVariance, for example). This means that the parameter object should contain a NamedTuple with keys pertaining to the distribution. 

Currently, we only support Gaussian distributions, meaning that the parameter object should contain ```(μ = ..., σ² = ..., )``` for mean-field or ```(μ = ..., Σ = ..., )``` for full-rank variational posteriors. The corresponding state object should contain an `epsilon` key that contains samples `ϵ ~ N(0, 1)` that are used to sample from the variational distribution.

The gradient with respect to μ can be estimated using the reparameterization trick, where we take a sample `z ~ N(μ, Σ)` and calculate the loss using the sample. This functionality is provided in the `sample_z(ps, st)` function, replacing the distribution parameters by `(z = ..., )` that can be used in the model definition to make predictions. The `epsilon` in `st` can be updated using `update_epsilon(rng, st)` to take a new sample from the variational distribution.

The gradient with respect to Σ can be estimated from ∇z using the `estimate_covariance_gradient_from_dz(∇z, ps, st)` function. This function calculates ∇Σ ≈ Σ⁻¹(z - μ) ∇zᵀ. Alternatives are to calculate the gradient with respect to the original parameter object (which contains Σ), or by ∇Σ ≈ 0.5 ⋅ ∇²z.

We can estimate the full gradient using a single sample of z or by taking multiple samples:

```julia
M = 3 # number of Monte Carlo samples
∇_m = map(1:M) do m
    update_epsilon!(rng, st)
    ps_z = sample_z(ps, st)
    ∇z = gradient(neg_logjoint, X, y, ps_z, st) # calculate gradient with respect to -p(x, z)
    return estimate_covariance_gradient_from_dz(∇z, ps, st)
end
∇ = fmap(Base.Fix2(/, M) ∘ +, ∇_m)
dlogq!(∇, ps) # Finally, we add the gradient wrt E_q[log q(z)]
```

Next we can update the parameters using the update function from Optimisers.jl:

```julia
opt_state, ps = Optimisers.update(opt_state, ps, ∇)
update_state!(opt_state, ps) # updates the state of NaturalDescentMean with the updated variance
```

### Next steps

* Adding "implicit" natural gradient descent, where the posterior variance is stored inside of the optimizer. This makes it easy to use the package with existing models.
* Adding natural momentum.
* Adding the IVON optimizer.
* Adding natural gradient descent for Cholesky based parameterizations.
* Supporting other distributions (SkewNormal, GaussianMixtures, ...).

### Relevant work

```
Khan, M. E., & Rue, H. (2023). The Bayesian learning rule. Journal of Machine Learning Research, 24(281), 1-46.
```

```
Lin, W., Schmidt, M., & Khan, M. E. (2020, November). Handling the positive-definite constraint in the Bayesian learning rule. In International conference on machine learning (pp. 6116-6126). PMLR.
```


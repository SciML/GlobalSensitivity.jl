# Regression Method

```@docs
RegressionGSA(; rank::Bool = false)
```

## Regression Details

It is possible to fit a linear model explaining the behavior of Y given the
values of X, provided that the sample size n is sufficiently large (at least n > d).

The measures provided for this analysis by us in GlobalSensitivity.jl are

  a) Pearson Correlation Coefficient:

```math
r = \frac{\sum_{i=1}^{n} (x_i - \overline{x})(y_i - \overline{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \overline{x})^2(y_i - \overline{y})^2}}
```

  b) Standard Regression Coefficient (SRC):

```math
SRC_j = \beta_{j} \sqrt{\frac{Var(X_j)}{Var(Y)}}
```

where ``\beta_j`` is the linear regression coefficient associated to $X_j$. This is also known
as a sigma-normalized derivative.

  c) Partial Correlation Coefficient (PCC):

```math
PCC_j = \rho(X_j - \hat{X_{-j}},Y_j - \hat{Y_{-j}})
```

where ``\hat{X_{-j}}`` is the prediction of the linear model, expressing ``X_{j}``
with respect to the other inputs and ``\hat{Y_{-j}}`` is the prediction of the
linear model where ``X_j`` is absent. PCC measures the sensitivity of ``Y`` to
``X_j`` when the effects of the other inputs have been canceled.

If `rank` is set to `true`, then the rank coefficients are also calculated.

### API

```@docs
gsa(f, method::RegressionGSA, p_range::AbstractVector; samples::Int = 1000, batch::Bool = false, kwargs...)
```

### Example

```julia
using GlobalSensitivity

function linear_batch(X)
    A= 7
    B= 0.1
    @. A*X[1,:]+B*X[2,:]
end
function linear(X)
    A= 7
    B= 0.1
    A*X[1]+B*X[2]
end

p_range = [[-1, 1], [-1, 1]]
reg = gsa(linear_batch, RegressionGSA(), p_range; batch = true)

reg = gsa(linear, RegressionGSA(), p_range; batch = false)
reg = gsa(linear, RegressionGSA(true), p_range; batch = false) #with rank coefficients
```

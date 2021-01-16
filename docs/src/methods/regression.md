# Regression Method

```julia
struct RegressionGSA <: GSAMethod
    rank::Bool = false
end
```

`RegressionGSA` has the following keyword arguments:

- `rank`: flag which determines whether to calculate the rank coefficients. Defaults to `false`.

It returns a `RegressionGSAResult`, which contains the `pearson`, `standard_regression`, and
`partial_correlation` coefficients, described below. If `rank` is true, then it also contains the ranked
versions of these coefficients. Note that the ranked version of the `pearson` coefficient is
also known as the Spearman coefficient, which is returned here as the `pearson_rank` coefficient.

For multi-variable models, the coefficient for the `` X_i `` input variable relating to the
`` Y_j `` output variable is given as the `[i, j]` entry in the corresponding returned matrix.

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

`function gsa(f, method::RegressionGSA, p_range::AbstractVector; samples::Int = 1000, batch::Bool = false, kwargs...)`

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
@doc raw"""

    RegressionGSA(; rank::Bool = false)

- `rank::Bool = false`: Flag determining whether to also run a rank regression analysis

Providing this to `gsa` results in a calculation of the following statistics, provided as a `RegressionGSAResult`. If
the function `f` to be analyzed is of dimensionality ``f: R^n -> R^m``, then these coefficients
are returned as a matrix, with the corresponding statistic in the `(i, j)` entry.

- `pearson`: This is equivalent to the correlation coefficient matrix between input and output. The rank version is known as the Spearman coefficient.
- `standard_regression`: Standard regression coefficients, also known as sigma-normalized derivatives
- `partial_correlation`: Partial correlation coefficients, related to the precision matrix and a measure of the correlation of linear models of the

## Method Details

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

where ``\beta_j`` is the linear regression coefficient associated to ``X\_j``. This is also known
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

## API

    gsa(f, method::RegressionGSA, p_range::AbstractVector; samples::Int, batch = false)
    gsa(X, Y, method::RegressionGSA)

### Example

```julia
using GlobalSensitivity

function linear_batch(X)
    A = 7
    B = 0.1
    @. A * X[1, :] + B * X[2, :]
end
function linear(X)
    A = 7
    B = 0.1
    A * X[1] + B * X[2]
end

p_range = [[-1, 1], [-1, 1]]
reg = gsa(linear_batch, RegressionGSA(), p_range; batch = true)

reg = gsa(linear, RegressionGSA(), p_range; batch = false)
reg = gsa(linear, RegressionGSA(true), p_range; batch = false) #with rank coefficients

X = QuasiMonteCarlo.sample(1000, [-1, -1], [1, 1], QuasiMonteCarlo.SobolSample())
Y = reshape(linear.([X[:, i] for i in 1:1000]), 1, 1000)
reg_mat = gsa(X, Y, RegressionGSA(true))
```
"""
struct RegressionGSA <: GSAMethod
    rank::Bool
end

RegressionGSA(; rank::Bool = false) = RegressionGSA(rank)

struct RegressionGSAResult{T, TR}
    pearson::T
    standard_regression::T
    partial_correlation::T
    pearson_rank::TR
    standard_rank_regression::TR
    partial_rank_correlation::TR
end

function gsa(X::AbstractArray, Y::AbstractArray, method::RegressionGSA)
    srcs = _calculate_standard_regression_coefficients(X, Y)
    corr = _calculate_correlation_matrix(X, Y)
    partials = _calculate_partial_correlation_coefficients(X, Y)

    if method.rank
        X_rank = vcat((sortperm(view(X, i, :))' for i in axes(X, 1))...)
        Y_rank = vcat((sortperm(view(Y, i, :))' for i in axes(Y, 1))...)

        srcs_rank = _calculate_standard_regression_coefficients(X_rank, Y_rank)
        corr_rank = _calculate_correlation_matrix(X_rank, Y_rank)
        partials_rank = _calculate_partial_correlation_coefficients(X_rank, Y_rank)

        return RegressionGSAResult(corr,
            srcs,
            partials,
            corr_rank,
            srcs_rank,
            partials_rank)
    end

    return RegressionGSAResult(corr,
        srcs,
        partials,
        nothing, nothing, nothing)
end

function _calculate_standard_regression_coefficients(X, Y)
    β̂ = X' \ Y'
    srcs = (β̂ .* std(X, dims = 2) ./ std(Y, dims = 2)')
    return Matrix(transpose(srcs))
end

function _calculate_correlation_matrix(X, Y)
    corr = cov(X, Y, dims = 2) ./ (std(X, dims = 2) .* std(Y, dims = 2)')
    return Matrix(transpose(corr))
end

function _calculate_partial_correlation_coefficients(X, Y)
    XY = vcat(X, Y)
    corr = cov(XY, dims = 2) ./ (std(XY, dims = 2) .* std(XY, dims = 2)')
    prec = pinv(corr) # precision matrix
    pcc_XY = -prec ./ sqrt.(diag(prec) .* diag(prec)')
    # return partial correlation matrix relating f: X -> Y model values
    return Matrix(transpose(pcc_XY[axes(X, 1), lastindex(X, 1) .+ axes(Y, 1)]))
end

function gsa(f, method::RegressionGSA, p_range::AbstractVector; samples::Int, batch = false)
    lb = [i[1] for i in p_range]
    ub = [i[2] for i in p_range]
    X = QuasiMonteCarlo.sample(samples, lb, ub, QuasiMonteCarlo.SobolSample())
    desol = false

    if batch
        _y = f(X)
        multioutput = _y isa AbstractMatrix
        Y = multioutput ? _y : reshape(_y, 1, length(_y))
    else
        _y = [f(X[:, j]) for j in axes(X, 2)]
        multioutput = !(eltype(_y) <: Number)
        if eltype(_y) <: RecursiveArrayTools.AbstractVectorOfArray
            y_size = size(_y[1])
            _y = vec.(_y)
            desol = true
        end
        Y = multioutput ? reduce(hcat, _y) : reshape(_y, 1, length(_y))
    end

    return gsa(X, Y, method)
end

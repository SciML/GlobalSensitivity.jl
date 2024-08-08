@doc raw"""

MutualInformation(; n_bootstraps = 1000, conf_level = 0.95)

- `n_bootstraps`: Number of bootstraps to be used for estimation of null distribution. Default is `1000`.
- `conf_level`: Confidence level for the minimum bound estimation. Default is `0.95`.


## Method Details

The sensitivity analysis based on mutual information is an alternative approach to sensitivity analysis based on information theoretic measures. In this method,
the output uncertainty is quantified by the entropy of the output distribution, instead of taking a variance-based approach. The Shannon entropy of the output is 
given by:

```math
H(Y) = -\sum_y p(y) \log p(y)
```
Where ``p(y)`` is the probability density function of the output ``Y``. By fixing an input ``X_i``, the conditional entropy of the output ``Y`` is given by:

```math
H(Y|X_i) = -\sum_{x} p(x) H(Y|X_i = x)
```

The mutual information between the input ``X_i`` and the output ``Y`` is then given by:

```math
I(X_i;Y) = H(Y) - H(Y|X_i) = H(X) + H(Y) - H(X,Y)
```

Where ``H(X,Y)`` is the joint entropy of the input and output. The mutual information can be used to calculate the sensitivity indices of the input parameters. 

### Sensitivity Indices
The sensitivity indices are calculated as the mutual information between the input ``X_i`` and the output ``Y`` where the ``\alpha`` quantile of
null distribution of the output is subtracted from the mutual information:

```math
S_{1,i} = I(X_i;Y) - Q(I(X_i; Y_\text{null}), \alpha)
```

Using mutual information for sensitivity analysis is introduced in Lüdtke et al. (2007)[^1] and also present in Datseris & Parlitz (2022)[^2]. 

## API

    gsa(f, method::MutualInformation, p_range; samples, batch = false)

Returns a `MutualInformationResult` object containing the resulting sensitivity indices for the parameters and the corresponding confidence intervals.
The `MutualInformationResult` object contains the following fields:
- `S`: Sensitivity indices.
- `mutual_information`: Computed mutual information values
- `bounds`: Computed upper bounds of the null distribution of mutual information.

### Example

```julia
using GlobalSensitivity

function ishi_batch(X)
    A= 7
    B= 0.1
    @. sin(X[1,:]) + A*sin(X[2,:])^2+ B*X[3,:]^4 *sin(X[1,:])
end
function ishi(X)
    A= 7
    B= 0.1
    sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1])
end

lb = -ones(4)*π
ub = ones(4)*π

res1 = gsa(ishi,MutualInformation(),[[lb[i],ub[i]] for i in 1:4],samples=15000)
res2 = gsa(ishi_batch,MutualInformation(),[[lb[i],ub[i]] for i in 1:4],samples=15000,batch=true)
```

### References
[^1]: Lüdtke, N., Panzeri, S., Brown, M., Broomhead, D. S., Knowles, J., Montemurro, M. A., & Kell, D. B. (2007). Information-theoretic sensitivity analysis: a general method for credit assignment in complex networks. Journal of The Royal Society Interface, 5(19), 223–235.
[^2]: Datseris, G., & Parlitz, U. (2022). Nonlinear Dynamics, Ch. 7, pg. 105-119.
"""
struct MutualInformation <: GSAMethod
    n_bootstraps::Int
    conf_level::Real
end

function MutualInformation(; n_bootstraps = 1000, conf_level = 0.95)
    MutualInformation(n_bootstraps, conf_level)
end

struct MutualInformationResult{T}
    S::T
    mutual_information::T
    bounds::T
end

function _compute_bounds(Xi, Y, conf_level, n_bootstraps)

    # perform permutations of Y and calculate mutual information
    mi_values = zeros(n_bootstraps)
    est = ValueHistogram(Int(round(sqrt(length(Y)))))
    Y_perm = copy(Y)
    entropy_Xi = entropy(est, Xi)
    entropy_Y = entropy(est, Y)
    for i in 1:n_bootstraps
        shuffle!(Y_perm)
        mi_values[i] = entropy_Xi + entropy_Y -
                             entropy(est, StateSpaceSet(Xi, Y_perm))
    end

    return quantile(mi_values, conf_level)
end

function _compute_mi(X::AbstractArray, Y::AbstractVector, method::MutualInformation)
    # K is the number of variables, samples is the number of simulations
    K = size(X, 1)

    if method.n_bootstraps > size(X, 2)
        throw(ArgumentError("Number of bootstraps must be less than or equal to the number of samples"))
    end

    est = ValueHistogram(Int(round(sqrt(size(Y, 1)))))
    entropy_Y = entropy(est, Y)

    sensitivities = zeros(K)
    bounds = zeros(K)

    # calculate mutual information
    @inbounds for i in 1:K
        Xi = @view X[i, :]
        sensitivities[i] = entropy(est, Xi) + entropy_Y -
                                entropy(est, StateSpaceSet(Xi, Y))
        bounds[i] = _compute_bounds(Xi, Y, method.conf_level, method.n_bootstraps)
    end

    sensitivities, bounds
end

function gsa(f, method::MutualInformation, p_range; samples, batch = false)
    lb = [i[1] for i in p_range]
    ub = [i[2] for i in p_range]

    X = QuasiMonteCarlo.sample(samples, lb, ub, QuasiMonteCarlo.SobolSample())

    if batch
        Y = f(X)
        multioutput = Y isa AbstractMatrix
    else
        Y = [f(X[:, j]) for j in axes(X, 2)]
        multioutput = !(eltype(Y) <: Number)
        if eltype(Y) <: RecursiveArrayTools.AbstractVectorOfArray
            y_size = size(Y[1])
            Y = vec.(Y)
            desol = true
        end
    end

    mutual_informations, bounds = _compute_mi(X, Y, method)

    mi_sensitivity = max.(0.0, mutual_informations .- bounds)

    return MutualInformationResult(mi_sensitivity, mutual_informations, bounds)
end

@doc raw"""

    RSA(; n_dummy_parameters::Int = 10, acceptance_threshold::Union{Function, Real} = mean)

- `n_dummy_parameters`: Number of dummy parameters to add to the model, used for sensitivity hypothesis testing and to check the amount of samples. Defaults to 10.
- `acceptance_threshold`: Threshold or function to compute the threshold for defining the acceptance distribution of the sensitivity outputs. The function must be of signature f(Y) 
   and return a real number, where Y is the output of given sensitivity criterion. Defaults to the mean of the sensitivity values.  

## Method Details

The RSA (Regional Sensitivity Analysis) method[^1] is a monte-carlo based technique for performing nonparametric global sensitivity analysis. The method is based on the Kolmogorov-Smirnov (KS) test, which is a nonparametric test of the equality of continuous, 
one-dimensional probability distributions. Each result of a monte-carlo simulation is either classified as behavior (``B``) or non-behavior (``\bar{B}``). For each parameter ``i``, 
the cumulative distributions of the behavior and non-behavior outputs are determined as ``F_{i}`` and ``\bar{F}_{i}``, respectively. The sensitivity for each parameter ``i`` is then given by:

```math
S_{i} = \sup_{j} |F_{i,j} - (1 - F_{i,j})|
```

where ``F_{i,j}`` is sample ``j`` of the cumulative distribution of the sensitivity output for parameter ``i``.

Dummy parameters are added to the model to check the amount of samples and to perform sensitivity hypothesis testing. Note that because of random sampling the results may vary between runs. 

## API

    gsa(f, method::RSA, p_range; samples, batch = false)

Returns a `RSAResult` object containing the sensitivity indices for the parameters as `S`, and mean and standard deviation of the dummy parameters as a tuple `Sd = (<mean>, <std>)`.

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

res1 = gsa(ishi,RSA(),[[lb[i],ub[i]] for i in 1:4],samples=15000)
res2 = gsa(ishi_batch,RSA(),[[lb[i],ub[i]] for i in 1:4],samples=15000,batch=true)
```

### References
[^1]: Hornberger, G.M. & Spear, Robert. (1981). An Approach to the Preliminary Analysis of Environmental Systems. J. Environ. Manage.; (United States). 12:1. 
"""
struct RSA <: GSAMethod
    n_dummy_parameters::Int
    acceptance_threshold::Union{Function, Real}
end

function RSA(; n_dummy_parameters = 10, acceptance_threshold = mean)
    return RSA(n_dummy_parameters, acceptance_threshold)
end

struct RSAResult{T}
    S::AbstractVector{T}
    Sd::Tuple{T, T}
end

function rsa_sensitivity(Xi, flag)
    sort_idx = sortperm(Xi)
    acceptance_dist = cumsum(flag[sort_idx])
    rejection_dist = cumsum(1 .- flag[sort_idx])

    # normalize distributions
    acceptance_dist = acceptance_dist / maximum(acceptance_dist)
    rejection_dist = rejection_dist / maximum(rejection_dist)

    return maximum(abs.(acceptance_dist .- rejection_dist))
end

function _compute_rsa(X::AbstractArray, Y::AbstractArray, method::RSA)

    # K is the number of variables, samples is the number of simulations
    K = size(X, 1)
    #samples = size(X, 2)
    sensitivities = zeros(K)

    if method.acceptance_threshold isa Function
        acceptance_threshold = method.acceptance_threshold(Y)
    else
        acceptance_threshold = method.acceptance_threshold
    end
    flag = map!(>(acceptance_threshold), similar(Y), Y)

    # Cumulative distributions (for model parameters and dummies)
    @inbounds for i in 1:K
        Xi = @view X[i, :]

        # calculate KS score
        sensitivities[i] = rsa_sensitivity(Xi, flag)
    end

    # collect dummy sensitivities (mean and std)
    dummy_sensitivities = (
        mean(sensitivities[(K - method.n_dummy_parameters + 1):end]),
        std(sensitivities[(K - method.n_dummy_parameters + 1 + 1):end]),
    )

    return RSAResult(
        sensitivities[1:(K - method.n_dummy_parameters)], dummy_sensitivities
    )
end

function gsa(f, method::RSA, p_range; samples, batch = false)
    lb = [i[1] for i in p_range]
    ub = [i[2] for i in p_range]

    # add dummy parameters
    lb = [lb; repeat([typeof(lb[1])(0.0)], method.n_dummy_parameters)]
    ub = [ub; repeat([typeof(ub[1])(1.0)], method.n_dummy_parameters)]
    X = QuasiMonteCarlo.sample(samples, lb, ub, QuasiMonteCarlo.LatinHypercubeSample())

    X̂ = X[1:length(lb), :]
    if batch
        Y = f(X̂)
        multioutput = Y isa AbstractMatrix
    else
        Y = [f(X̂[:, j]) for j in axes(X̂, 2)]
        multioutput = !(eltype(Y) <: Number)
        if eltype(Y) <: RecursiveArrayTools.AbstractVectorOfArray
            y_size = size(Y[1])
            Y = vec.(Y)
            desol = true
        end
    end
    return _compute_rsa(X, Y, method)
end

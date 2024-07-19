@doc raw"""

    KSRank(; n_dummy_parameters::Int = 50, acceptance_threshold::Union{Function, Real} = mean)

- `n_dummy_parameters`: Number of dummy parameters to add to the model, used for sensitivity hypothesis testing and to check the amount of samples. Defaults to 50.
- `acceptance_threshold`: Threshold or function to compute the threshold for defining the acceptance distribution of the sensitivity outputs. The function must be of signature f(Y) 
   and return a real number, where Y is the output of given sensitivity criterion. Defaults to the mean of the sensitivity values.  

## Method Details

The KSRank method is a monte-carlo based technique for performing nonparametric global sensitivity analysis. The method is based on the Kolmogorov-Smirnov (KS) test, which is a nonparametric test of the equality of continuous, 
one-dimensional probability distributions that can be used to compare a sample with a reference probability distribution. The method is used to compare the acceptance and rejection distributions of the sensitivity outputs, which are
calculated by comparing the sensitivity outputs with a threshold value. The sensitivity for each parameter ``i`` is then given by

```math
S_{i} = \sup_{j} |F_{i,j} - (1 - F_{i,j})|
```

where ``F_{i,j}`` is sample ``j`` of the cumulative acceptance distribution of the sensitivity output for parameter ``i``.

Dummy parameters are added to the model to check the amount of samples and to perform sensitivity hypothesis testing. Note that because of random sampling the results may vary between runs. 

See also [Hornberger & Spear (1981)](https://www.researchgate.net/profile/Robert-Spear/publication/236357160_An_Approach_to_the_Preliminary_Analysis_of_Environmental_Systems/links/57a36cfa08aefe6167a599af/An-Approach-to-the-Preliminary-Analysis-of-Environmental-Systems.pdf)

## API

    gsa(f, method::KSRank, p_range; samples, batch = false)

Returns a `KSRankResult` object containing the sensitivity indices for the parameters as `S`, and mean and standard deviation of the dummy parameters as a tuple `Sd = (<mean>, <std>)`.

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

res1 = gsa(ishi,KSRank(),[[lb[i],ub[i]] for i in 1:4],samples=15000)
res2 = gsa(ishi_batch,KSRank(),[[lb[i],ub[i]] for i in 1:4],samples=15000,batch=true)
```
"""
struct KSRank <: GSAMethod
    n_dummy_parameters::Int
    acceptance_threshold::Union{Function, Real}
end

function KSRank(; n_dummy_parameters = 50, acceptance_threshold = mean)
    KSRank(n_dummy_parameters, acceptance_threshold)
end

struct KSRankResult{T}
    S::AbstractVector{T}
    Sd::Tuple{T, T}
end

function ks_rank_sensitivity(Xi, flag)
    sort_idx = sortperm(Xi)
    acceptance_dist = cumsum(flag[sort_idx])
    rejection_dist = cumsum(1 .- flag[sort_idx])

    # normalize distributions
    acceptance_dist = acceptance_dist / maximum(acceptance_dist)
    rejection_dist = rejection_dist / maximum(rejection_dist)

    maximum(abs.(acceptance_dist .- rejection_dist))
end

function _compute_ksrank(X::AbstractArray, Y::AbstractArray, method::KSRank)

    # K is the number of variables, samples is the number of simulations
    K = size(X, 1)
    #samples = size(X, 2)
    sensitivities = zeros(K)

    if method.acceptance_threshold isa Function
        acceptance_threshold = method.acceptance_threshold(Y)
    else
        acceptance_threshold = method.acceptance_threshold
    end
    flag = Int.(Y .> acceptance_threshold)

    # Cumulative distributions (for model parameters and dummies)
    @inbounds for i in 1:K
        Xi = @view X[i, :]

        # calculate KS score
        sensitivities[i] = ks_rank_sensitivity(Xi, flag)
    end

    # collect dummy sensitivities (mean and std)
    dummy_sensitivities = (mean(sensitivities[(K - method.n_dummy_parameters + 1):end]),
        std(sensitivities[(K - method.n_dummy_parameters + 1 + 1):end]))

    return KSRankResult(
        sensitivities[1:(K - method.n_dummy_parameters)], dummy_sensitivities)
end

function gsa(f, method::KSRank, p_range; samples, batch = false)
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
    return _compute_ksrank(X, Y, method)
end

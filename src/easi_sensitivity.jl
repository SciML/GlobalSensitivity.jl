"""

    EASI(; max_harmonic::Int = 10, dct_method::Bool = false)

- `max_harmonic`: Maximum harmonic of the input frequency for which the output power spectrum is analyzed for. Defaults to 10.
- `dct_method`: Use Discrete Cosine Transform method to compute the power spectrum. Defaults to false.

## Method Details

The EASI method is a Fourier-based technique for performing
variance-based methods of global sensitivity analysis for the
computation of first order effects (Sobol' indices), hence belonging
into the same class of algorithms as FAST and RBD. It is a
computationally cheap method for which existing data can be used.
Unlike the FAST and RBD methods which use a specially generated sample
set that contains suitable frequency data for the input factors, in
EASI these frequencies are introduced by sorting and shuffling the
available input samples.

## API

    gsa(f, method::EASI, p_range; samples, batch = false)
    gsa(X, Y, method::EASI)

### Example

```julia
using GlobalSensitivity, Test

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

res1 = gsa(ishi,EASI(),[[lb[i],ub[i]] for i in 1:4],samples=15000)
res2 = gsa(ishi_batch,EASI(),[[lb[i],ub[i]] for i in 1:4],samples=15000,batch=true)

X = QuasiMonteCarlo.sample(15000, lb, ub, QuasiMonteCarlo.SobolSample())
Y = ishi.([X[:, i] for i in 1:15000])

res1 = gsa(X, Y, EASI())
res1 = gsa(X, Y, EASI(; dct_method = true))
```
"""
struct EASI <: GSAMethod
    max_harmonic::Int
    dct_method::Bool
end

EASI(; max_harmonic = 4, dct_method = false) = EASI(max_harmonic, dct_method)

struct EASIResult{T}
    S1::T
    S1_Corr::T
end

"""
Code based on the theory presented in
    - Elmar Plischke (2010) "An effective algorithm for computing global
      sensitivity indices (EASI) Reliability Engineering & System Safety",
      95:4, 354-360. doi:10.1016/j.ress.2009.11.005

and the python implementation of EASI in python's Sensitivty Analysis Library ("SALib")
"""

function _permute_outputs(X::AbstractArray, Y::AbstractArray)
    """
    Triangular shape permutation of the precomputed inputs
    """
    permutation_index = sortperm(X) # non-mutating
    result = cat(permutation_index[1:2:end], reverse(permutation_index[2:2:end]), dims = 1)
    return @view Y[result]
end

function _compute_first_order_fft(permuted_outputs, max_harmonic, samples)
    ft = (fft(permuted_outputs))[2:(samples ÷ 2)]
    ys = abs2.(ft) .* inv(samples)
    V = 2 * sum(ys)
    Vi = 2 * sum(ys[(1:max_harmonic)])
    Si = Vi / V
end

"""
Elmar Plischke,
How to compute variance-based sensitivity indicators with your spreadsheet software,
Environmental Modelling & Software,
Volume 35,
2012,
Pages 188-191,
ISSN 1364-8152,
https://doi.org/10.1016/j.envsoft.2012.03.004.
"""
function _compute_first_order_dct(permuted_outputs, max_harmonic, samples)
    ft = dct(permuted_outputs)[2:end]
    V = sum(abs2, ft)
    Vi = sum(abs2, ft[(1:max_harmonic)])
    Si = Vi / V
end

function _unskew_S1(S1::Number, max_harmonic::Integer, samples::Integer)
    """
    Unskew the sensivity index
    (Jean-Yves Tissot, Clémentine Prieur (2012) "Bias correction for the
    estimation of sensitivity indices based on random balance designs.",
    Reliability Engineering and System Safety, Elsevier, 107, 205-213.
    doi:10.1016/j.ress.2012.06.010)
    """
    λ = (2 * max_harmonic) / samples
    return S1 - (λ / (1 - λ)) * (1 - S1)
end

function gsa(X, Y, method::EASI)

    # K is the number of variables, samples is the number of simulations
    K = size(X, 1)
    samples = size(X, 2)
    sensitivites = zeros(K)
    sensitivites_c = zeros(K)

    for i in 1:K
        Xi = @view X[i, :]

        if method.dct_method
            S1 = _compute_first_order_dct(Y[sortperm(Xi)], method.max_harmonic, samples)
        else
            Y_reordered = _permute_outputs(Xi, Y)
            S1 = _compute_first_order_fft(Y_reordered, method.max_harmonic, samples)
        end

        S1_C = _unskew_S1(S1, method.max_harmonic, samples) # get bias-corrected version
        sensitivites[i] = S1
        sensitivites_c[i] = S1_C
    end

    return EASIResult(sensitivites, sensitivites_c)
end

function gsa(f, method::EASI, p_range; samples, batch = false)
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
    return gsa(X, Y, method)
end

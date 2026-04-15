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

and the python implementation of EASI in python's Sensitivity Analysis Library ("SALib")
"""

function _gather_triangular_permutation!(dest::AbstractVector, perm::Vector{<:Integer}, Y::AbstractArray)
    #=
    Apply the triangular-shape permutation (odd-indexed samples forward,
    even-indexed samples in reverse) to Y using the sort-order vector `perm`,
    writing the result directly into `dest`.
    Avoids intermediate allocations
    =#
    n = length(perm)
    j = firstindex(dest) - 1
    # Odd-indexed entries of perm (1, 3, 5, ...) in forward order
    for k in 1:2:n
        j += 1
        dest[j] = Y[perm[k]]
    end
    # Even-indexed entries of perm (2, 4, 6, ...) in reverse order
    last_even = 2 * (n ÷ 2)
    for k in last_even:-2:2
        j += 1
        dest[j] = Y[perm[k]]
    end
    return dest
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

function _unskew_S1(S1::Number, max_harmonic::Integer, samples::Integer)
    """
    Unskew the sensitivity index
    (Jean-Yves Tissot, Clémentine Prieur (2012) "Bias correction for the
    estimation of sensitivity indices based on random balance designs.",
    Reliability Engineering and System Safety, Elsevier, 107, 205-213.
    doi:10.1016/j.ress.2012.06.010)
    """
    λ = (2 * max_harmonic) / samples
    return S1 - (λ / (1 - λ)) * (1 - S1)
end

function gsa(X::AbstractArray, Y::AbstractArray, method::EASI)
    # K is the number of variables, samples is the number of simulations
    K = size(X, 1)
    samples = size(X, 2)

    # Reuseable buffers for sorting
    Yperm = Matrix{eltype(Y)}(undef, samples, K)
    perm = Vector{Int}(undef, samples)

    sensitivities = if method.dct_method
        for (i, xi) in zip(axes(Yperm, 2), eachrow(X))
            sortperm!(perm, xi)
            for (k, pk) in zip(axes(Yperm, 1), perm)
                Yperm[k, i] = Y[pk]
            end
        end
        dct_Yperm = dct(Yperm, 1)
        sensitivities = let max_harmonic = method.max_harmonic
            map(eachcol(dct_Yperm)) do dcti
                return sum(abs2, @view(dcti[(begin + 1):min(begin + max_harmonic, end)])) / sum(abs2, @view(dcti[(begin + 1):end]))
            end
        end
    else
        for (i, xi) in zip(axes(Yperm, 2), eachrow(X))
            sortperm!(perm, xi)
            _gather_triangular_permutation!(@view(Yperm[:, i]), perm, Y)
        end
        rfft_Yperm = rfft(Yperm, 1)
        sensitivities = let max_harmonic = method.max_harmonic
            map(eachcol(rfft_Yperm)) do rffti
                return sum(abs2, @view(rffti[(begin + 1):min(begin + max_harmonic, end - 1)])) / sum(abs2, @view(rffti[(begin + 1):(end - 1)]))
            end
        end
    end

    sensitivities_c = map(s -> _unskew_S1(s, method.max_harmonic, samples), sensitivities)

    return EASIResult(sensitivities, sensitivities_c)
end

function gsa(f, method::EASI, p_range; samples, batch = false)
    lb = [float(i[1]) for i in p_range]
    ub = [float(i[2]) for i in p_range]
    X = QuasiMonteCarlo.sample(samples, lb, ub, QuasiMonteCarlo.SobolSample())

    Y = if batch
        f(X)
    else
        [(y = f(x); y isa RecursiveArrayTools.AbstractVectorOfArray ? vec(y) : y) for x in eachcol(X)]
    end

    return gsa(X, Y, method)
end

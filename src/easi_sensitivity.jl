struct EASI <: GSAMethod
    max_harmonic::Int
    dct_method::Bool
end
EASI(; max_harmonic=4, dct_method=false) = EASI(max_harmonic, dct_method)

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
    result = cat(permutation_index[1:2:end], reverse(permutation_index[2:2:end]), dims=1)
    return @view Y[result]
end


function _compute_first_order_fft(permuted_outputs, max_harmonic, N)
    ft = (fft(permuted_outputs))[2:(N÷2)]
    ys = abs2.(ft) .* inv(N)
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
function _compute_first_order_dct(permuted_outputs, max_harmonic, N)
    ft = dct(permuted_outputs)[2:end]
    V = sum(abs2, ft)
    Vi = sum(abs2, ft[(1:max_harmonic)])
    Si = Vi / V
end

function _unskew_S1(S1::Number, max_harmonic::Integer, N::Integer)
    """
    Unskew the sensivity index
    (Jean-Yves Tissot, Clémentine Prieur (2012) "Bias correction for the
    estimation of sensitivity indices based on random balance designs.",
    Reliability Engineering and System Safety, Elsevier, 107, 205-213.
    doi:10.1016/j.ress.2012.06.010)
    """
    λ = (2 * max_harmonic) / N
    return S1 - (λ / (1 - λ)) * (1 - S1)
end


function gsa(f, method::EASI, p_range; N, batch=false, rng::AbstractRNG=Random.default_rng(), kwargs...)
    lb = [i[1] for i in p_range]
    ub = [i[2] for i in p_range]
    X = QuasiMonteCarlo.sample(N, lb, ub, QuasiMonteCarlo.SobolSample())

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

    # K is the number of variables, N is the number of simulations
    K = size(X, 1)
    sensitivites = zeros(K)
    sensitivites_c = zeros(K)

    for i in 1:K
        Xi = @view X[i, :]

        if method.dct_method
            S1 = _compute_first_order_dct(Y[sortperm(Xi)], method.max_harmonic, N)
        else
            Y_reordered = _permute_outputs(Xi, Y)
            S1 = _compute_first_order_fft(Y_reordered, method.max_harmonic, N)
        end

        S1_C = _unskew_S1(S1, method.max_harmonic, N) # get bias-corrected version
        sensitivites[i] = S1
        sensitivites_c[i] = S1_C
    end

    return EASIResult(sensitivites, sensitivites_c)
end

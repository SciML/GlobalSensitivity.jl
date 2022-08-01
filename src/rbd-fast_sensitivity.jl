struct RBDFAST <: GSAMethod
    num_harmonics::Int
end

"""
    RBDFAST(; num_harmonics = 6)

- num_harmonics: Number of harmonics to consider during power spectral density analysis.
"""
RBDFAST(; num_harmonics = 6) = RBDFAST(num_harmonics)

"""
Code based on the theory presented in:
    Saltelli, A. (2008). Global sensitivity analysis: The primer. Chichester: Wiley, pp. 167-169.
and
    S. Tarantola, D. Gatelli and T. Mara (2006)
    "Random Balance Designs for the Estimation of First Order Global Sensitivity Indices",
    Reliability Engineering and System Safety, 91:6, 717-727
"""

using FFTW, Random, Statistics, StatsBase, Distributions
allsame(x) = all(y -> y == first(x), x)

function gsa(f, method::RBDFAST; num_params, samples,
             rng::AbstractRNG = Random.default_rng(), batch = false, kwargs...)
    # Initalize matrix containing range of values of the parametric variable
    # along each column (factor).
    s0 = range(-π, stop = π, length = samples)

    # Compute inputs
    s = [s0[randperm(rng, samples)] for i in 1:num_params]
    x = hcat([0.5 .+ asin.(sin.(s[i])) ./ pi for i in 1:num_params]...)

    # Compute outputs
    if batch
        Y = f(x')
    else
        Y = [f(@view x[i, :]) for i in axes(x, 1)]
    end
    # Iterate over factors

    sensitivites = zeros(num_params)
    for i in 1:num_params
        s_order = sortperm(s[i])
        # Order Ys by how they would occur if they were
        # monotonically increasing as the
        # parametric variable s (not its permutation) increased.
        y_reordered = @view Y[s_order]

        ft = fft(y_reordered)
        ys = abs2.(ft) ./ samples
        V = sum(ys[2:samples])
        Vi = 2 * sum(ys[2:(method.num_harmonics + 1)])
        Si = Vi / V
        # println(ys)
        # unskew the sensitivies
        # lambda = 2*method.num_harmonics/samples
        # sensitivites[i] = Si - (lambda / (1 - lambda)) * (1-Si)
        sensitivites[i] = Si
    end

    return sensitivites
end

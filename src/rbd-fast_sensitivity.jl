"""

    RBDFAST(; num_harmonics = 6)

- `num_harmonics`: Number of harmonics to consider during power spectral density analysis.

## Method Details

In the Random Balance Designs (RBD) method, similar to `eFAST`,  `samples`
points are selected over a curve in the input space. A fixed frequency
equal to `1` is used for each factor. Then independent random
permutations are applied to the coordinates of the samples points in order to
generate the design points. The input model for analysis is evaluated
at each design point, and the outputs are reordered such that the design
points are in increasing order with respect to factor `Xi`. The Fourier
spectrum is calculated on the model output at the frequency 1 and at
its higher harmonics (2, 3, 4, 5, 6) and yields the estimate of the
sensitivity index of factor `Xi`.

## API

    gsa(f, method::RBDFAST; num_params, samples,
             rng::AbstractRNG = Random.default_rng(), batch = false, kwargs...)

### Example

```julia
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

lb = -ones(4)*π
ub = ones(4)*π

rng = StableRNG(123)
res1 = gsa(linear,GlobalSensitivity.RBDFAST(),num_params = 4, samples=15000)
res2 = gsa(linear_batch,GlobalSensitivity.RBDFAST(),num_params = 4, batch=true, samples=15000)
```
"""
struct RBDFAST <: GSAMethod
    num_harmonics::Int
end

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

function gsa(
        f, method::RBDFAST; num_params, samples,
        rng::AbstractRNG = Random.default_rng(), batch = false, kwargs...
    )
    # Initialize matrix containing range of values of the parametric variable
    # along each column (factor).
    s0 = range(-π, stop = π, length = samples)

    # Compute inputs
    s = [s0[randperm(rng, samples)] for i in 1:num_params]
    x = hcat([0.5 .+ asin.(sin.(s[i])) ./ pi for i in 1:num_params]...)

    # Compute outputs (type unstable, therefore we use a function barrier below)
    Y = if batch
        f(x')
    else
        [f(xj) for xj in eachrow(x)]
    end

    return _rbdfast_analysis(Y, s, method)
end

function _rbdfast_analysis(Y::AbstractVector{<:Real}, s::Vector{<:AbstractVector{<:Real}}, method::RBDFAST)
    # Build a (samples, num_params) matrix of per-parameter-reordered outputs:
    # each column holds Y reordered so the corresponding parametric variable
    # would increase monotonically.
    samples = length(Y)
    num_params = length(s)
    Yperm = Matrix{eltype(Y)}(undef, samples, num_params)
    perm = Vector{Int}(undef, samples)
    for (i, si) in zip(axes(Yperm, 2), s)
        sortperm!(perm, si)
        for (k, pk) in zip(axes(Yperm, 1), perm)
            Yperm[k, i] = Y[pk]
        end
    end
    ft = rfft(Yperm, 1)

    sensitivities = let H = method.num_harmonics
        map(eachcol(ft)) do fti
            V = 2 * sum(abs2, @view(fti[(begin + 1):(end - 1)])) + abs2(fti[end])
            Vi = 2 * sum(abs2, @view(fti[(begin + 1):(begin + H)]))
            return Vi / V
        end
    end

    return sensitivities
end

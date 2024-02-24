"""

    DeltaMoment(; nboot = 500, conf_level = 0.95, Ygrid_length = 2048,
                     num_classes = nothing)

- `nboot`: number of bootstrap repetitions. Defaults to `500`.
- `conf_level`: the level used for confidence interval calculation with bootstrap. Default value of `0.95`.
- `Ygrid_length`: number of quadrature points to consider when performing the kernel density estimation and the integration steps. Should be a power of 2 for efficient FFT in kernel density estimates. Defaults to `2048`.
- `num_classes`: Determine how many classes to split each factor into to when generating distributions of model output conditioned on class.

## Method Details

The Delta moment-independent method relies on new estimators for
density-based statistics.  It allows for the estimation of both
distribution-based sensitivity measures and of sensitivity measures that
look at contributions to a specific moment. One of the primary advantage
of this method is the independence of computation cost from the number of
parameters.

!!! note
    `DeltaMoment` only works for scalar output.

## API

    gsa(f, method::DeltaMoment, p_range; samples, batch = false,
             rng::AbstractRNG = Random.default_rng())
    gsa(X, Y, method::DeltaMoment; rng::AbstractRNG = Random.default_rng())

### Example

```julia
using GlobalSensitivity, Test

function ishi(X)
    A= 7
    B= 0.1
    sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1])
end

lb = -ones(3)*π
ub = ones(3)*π

m = gsa(ishi,DeltaMoment(),fill([lb[1], ub[1]], 3), samples=1000)


samples = 1000
X = QuasiMonteCarlo.sample(samples, lb, ub, QuasiMonteCarlo.SobolSample())
Y = ishi.(@view X[:, i] for i in 1:samples)

m = gsa(X, Y, DeltaMoment())
```
"""
struct DeltaMoment{T} <: GSAMethod
    nboot::Int
    conf_level::Float64
    Ygrid_length::Int
    num_classes::T
end

function DeltaMoment(; nboot = 500, conf_level = 0.95, Ygrid_length = 2048,
        num_classes = nothing)
    DeltaMoment(nboot, conf_level, Ygrid_length, num_classes)
end

struct DeltaResult{T}
    deltas::T
    adjusted_deltas::T
    adjusted_deltas_low::T
    adjusted_deltas_hi::T
end
function _calc_delta(Xi, Y, Ygrid, class_cutoffs)

    # Make sure Ys are not identical, otherwise KDE will be undefined.
    # If Ys are identical, then we know X and Y are independent, so return 0
    if all(Y[1] .== Y)
        return 0
    end

    samples = length(Y) # Number of simulations
    @assert length(Xi) == samples # Length of Y should equal length of X

    # Model pdf of Y using KDE, kde uses normal kernel by default
    fy = pdf(kde(Y), Ygrid) # eq 23.1

    # Get probability of each y in Ygrid.
    x_rank = competerank(Xi) # Does what scipy.stats rankdata does. If tie, all tied values get same rank.
    d_hat = 0 # the delta estimator.

    # Iterate over each class
    weighted_class_seps = zeros(length(class_cutoffs) - 1)
    for j in 1:(length(class_cutoffs) - 1)
        # get X and Y indices for samples that are in this class (where
        # class designation is based on the X value)
        condition(x) = (x > class_cutoffs[j]) == (x <= class_cutoffs[j + 1])
        in_class_indices = findall(condition, x_rank)
        number_in_class = length(in_class_indices)
        # Get the the subset of Y values in this class
        in_class_Y = Y[in_class_indices]
        if length(in_class_Y) == 0
            continue
        end
        # get the separation between the total y pdf and the condition y pdf induced by this class
        fyc = pdf(kde(in_class_Y), Ygrid) # eq 23.2 - Estimated conditional distribution of y (using normal kernel)
        pdf_diff = abs.(fy .- fyc) # eq 24
        # Use trapezoidal rule to estimate the difference between the curves.
        class_separation = trapz(Ygrid, pdf_diff) # eq 25
        # Increment estimator
        weighted_class_seps[j] = number_in_class * class_separation # eq 26
    end

    d_hat = sum(weighted_class_seps) / (2 * samples)
    return d_hat
end

function gsa(X::AbstractArray, Y::AbstractArray, method::DeltaMoment;
        rng::AbstractRNG = Random.default_rng())
    samples = size(X, 2)
    # Create number of classes and class cutoffs.
    if method.num_classes === nothing
        exp = (2 / (7 + tanh((1500 - samples) / 500)))
        M = Integer(round(min(Integer(ceil(samples^exp)), 48))) # Number of classes
    else
        M = method.num_classes
    end
    class_cutoffs = range(0, samples, length = M + 1) # class cutoffs

    # quadrature points.
    # Length should be a power of 2 for efficient FFT in kernel density estimates.
    Ygrid = range(minimum(Y), maximum(Y), length = method.Ygrid_length)

    num_factors = size(X, 1)

    deltas = zeros(num_factors)
    adjusted_deltas = zeros(num_factors)
    adjusted_deltas_conf = zeros(num_factors)
    adjusted_deltas_low = zeros(num_factors)
    adjusted_deltas_high = zeros(num_factors)

    for factor_num in 1:num_factors
        Xi = view(X, factor_num, :)
        delta = _calc_delta(Xi, Y, Ygrid, class_cutoffs)
        deltas[factor_num] = delta
        # eq. 30, bias reduction via bootstrapping.
        bootstrap_deltas = zeros(method.nboot)
        r = rand(rng, 1:samples, method.nboot, samples)
        for i in 1:(method.nboot)
            r_i = r[i, :]
            bootstrap_deltas[i] = _calc_delta(Xi[r_i], Y[r_i], Ygrid, class_cutoffs)
        end
        adjusted_deltas[factor_num] = 2 * delta - mean(bootstrap_deltas)
        band = quantile(Normal(0.0, 1.0), 0.5 + method.conf_level / 2) *
               std(bootstrap_deltas) / (sqrt(method.nboot))
        adjusted_deltas_low[factor_num] = adjusted_deltas[factor_num] - band
        adjusted_deltas_high[factor_num] = adjusted_deltas[factor_num] + band
    end

    return DeltaResult(deltas, adjusted_deltas, adjusted_deltas_low, adjusted_deltas_high)
end

function gsa(f, method::DeltaMoment, p_range; samples, batch = false,
        rng::AbstractRNG = Random.default_rng())
    lb = [i[1] for i in p_range]
    ub = [i[2] for i in p_range]
    X = QuasiMonteCarlo.sample(samples, lb, ub, QuasiMonteCarlo.SobolSample())

    if batch
        Y = f(X)
        multioutput = Y isa AbstractMatrix
        if multioutput
            throw(ArgumentError("DeltaMoment sensitivity only supports scalar output functions"))
        end
    else
        Y = [f(X[:, j]) for j in axes(X, 2)]
        multioutput = !(eltype(Y) <: Number)
        if eltype(Y) <: RecursiveArrayTools.AbstractVectorOfArray
            y_size = size(Y[1])
            Y = vec.(Y)
            desol = true
        end
        if multioutput
            throw(ArgumentError("DeltaMoment sensitivity only supports scalar output functions"))
        end
    end
    return gsa(X, Y, method; rng)
end

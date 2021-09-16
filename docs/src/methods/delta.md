# Delta Moment-Independent method

```julia
struct DeltaMoment{T} <: GSAMethod
    nboot::Int
    conf_level::Float64
    Ygrid_length::Int
    num_classes::T
end
```

`DeltaMoment` has the following keyword arguments:
    - `nboot`: number of bootstrap repitions. Defaukts to `500`.
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

### API

```julia
function gsa(f, method::DeltaMoment, p_range; N, batch = false, rng::AbstractRNG = Random.default_rng(), kwargs...)
```

### Example

```julia
using GlobalSensitivity, Test

function ishi(X)
    A= 7
    B= 0.1
    sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1])
end

lb = -ones(4)*π
ub = ones(4)*π

m = gsa(ishi,DeltaMoment(),fill([lb[1], ub[1]], 3), N=1000)
```
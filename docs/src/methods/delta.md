# Delta Moment-Independent Method

```@docs
DeltaMoment(; nboot = 500, conf_level = 0.95, Ygrid_length = 2048,
                     num_classes = nothing)
```

## Method Details

The Delta moment-independent method relies on new estimators for
density-based statistics.  It allows for the estimation of both
distribution-based sensitivity measures and of sensitivity measures that
look at contributions to a specific moment. One of the primary advantage
of this method is the independence of computation cost from the number of
parameters.

!!! note
    `DeltaMoment` only works for scalar output.

### API

```docs
gsa(f, method::DeltaMoment, p_range; samples, batch = false, rng::AbstractRNG = Random.default_rng(), kwargs...)
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

m = gsa(ishi,DeltaMoment(),fill([lb[1], ub[1]], 3), samples=1000)
```

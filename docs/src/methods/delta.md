# Delta Moment-Independent Method

```@docs
DeltaMoment(; nboot = 500, conf_level = 0.95, Ygrid_length = 2048,
                     num_classes = nothing)
```

## API

```@docs
gsa(f, method::DeltaMoment, p_range; samples, batch = false, rng::AbstractRNG = Random.default_rng(), kwargs...)
```

# eFAST Method

```@docs
eFAST(; num_harmonics::Int = 4)
```

## API

```@docs
gsa(f, method::eFAST, p_range::AbstractVector; samples::Int, batch = false,
             distributed::Val{SHARED_ARRAY} = Val(false),
             rng::AbstractRNG = Random.default_rng(), kwargs...) where {SHARED_ARRAY}
```

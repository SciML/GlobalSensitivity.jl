# EASI Method

```@docs
EASI(; max_harmonic::Int = 10, dct_method::Bool = false)
```

## Method Details

The EASI method is a Fourier-based technique for performing
variance-based methods of global sensitivity analysis for the
computation of first order effects (Sobol’ indices), hence belonging
into the same class of algorithms as FAST and RBD. It is a
computationally cheap method for which existing data can be used.
Unlike the FAST and RBD methods which use a specially generated sample
set that contains suitable frequency data for the input factors, in
EASI these frequencies are introduced by sorting and shuffling the
available input samples.

### API

```@docs
gsa(f, method::EASI, p_range; samples, batch = false, rng::AbstractRNG = Random.default_rng(), kwargs...)
```

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

```

# Random Balance Design FAST Method

```julia
struct RBDFAST <: GSAMethod  
    num_harmonics::Int
end
```

`RBDFAST` has the following keyword arguments:

- `num_harmonics`: Number of harmonics to consider during power spectral density analysis.

## Method Details

In the Random Balance Designs (RBD) method, similar to `eFAST`,  `N`
points are selected over a curve in the input space. A fixed frequency 
equal to `1` is used for each factor. Then independent random 
permutations are applied to the coordinates of the N points in order to 
generate the design points. The input model for analysis is evaluated 
at each design point and the outputs are reordered such that the design 
points are in increasing order with respect to factor `Xi`. The Fourier 
spectrum is calculated on the model output at the frequency 1 and at 
its higher harmonics (2, 3, 4, 5, 6) and yields the estimate of the 
sensitivity index of factor `Xi`.

### API

```julia
function gsa(f, method::RBDFAST; num_params, N, rng::AbstractRNG = Random.default_rng(), batch = false, kwargs...)
```

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
res1 = gsa(linear,GlobalSensitivity.RBDFAST(),num_params = 4, N=15000)
res2 = gsa(linear_batch,GlobalSensitivity.RBDFAST(),num_params = 4, batch=true, N=15000)
```
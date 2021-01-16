# eFAST Method

```julia
struct eFAST <: GSAMethod
    num_harmonics::Int
end
```

The `eFAST` object has `num_harmonics` as the only field, which is the number of harmonics to sum in
the Fourier series decomposition, this defaults to 4.

## eFAST Method Details

eFAST offers a robust, especially at low sample size, and computationally efficient procedure to
get the first and total order indices as discussed in Sobol. It utilizes monodimensional Fourier decomposition
along a curve exploring the parameter space. The curve is defined by a set of parametric equations,
```math
x_{i}(s) = G_{i}(sin ω_{i}s), ∀ i=1,2 ,..., n,
```
where s is a scalar variable varying over the range ``-∞ < s < +∞``, ``G_{i}`` are transformation functions
and ``{ω_{i}}, ∀ i=1,2,...,n`` is a set of different (angular) frequencies, to be properly selected, associated with each factor.
For more details on the transformation used and other implementation details you can go through [ A. Saltelli et al.](http://dx.doi.org/10.1080/00401706.1999.10485594).

### API

```julia
function gsa(f, method::eFAST, p_range::AbstractVector; n::Int=1000, batch=false, distributed::Val{SHARED_ARRAY} = Val(false), kwargs...) where {SHARED_ARRAY}
```

### Example

Below we show use of `eFAST` on the Ishigami function.

```julia
using GlobalSensitivity, QuasiMonteCarlo

function ishi(X)
    A= 7
    B= 0.1
    sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1])
end

lb = -ones(4)*π
ub = ones(4)*π

res1 = gsa(ishi,eFAST(),[[lb[i],ub[i]] for i in 1:4],n=15000)

##with batching
function ishi_batch(X)
    A= 7
    B= 0.1
    @. sin(X[1,:]) + A*sin(X[2,:])^2+ B*X[3,:]^4 *sin(X[1,:])
end

res2 = gsa(ishi_batch,eFAST(),[[lb[i],ub[i]] for i in 1:4],n=15000,batch=true)

```

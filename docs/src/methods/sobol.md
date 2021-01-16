# Sobol Method

```julia
struct Sobol <: GSAMethod
    order::Vector{Int}
    nboot::Int
    conf_int::Float64
end
```

The `Sobol` object has as its fields the `order` of the indices to be estimated. 
- `order` - the order of the indices to calculate. Defaults to `[0,1]`, which means the
  Total and First order indices. Passing `2` enables calculation of the Second order indices as well.

For confidence interval calculation `nboot` should be specified for the number (>0) of bootstrap runs 
and `conf_int` for the confidence level, the default for which is `0.95`.

## Sobol Method Details

Sobol is a variance-based method and it decomposes the variance of the output of
the model or system into fractions which can be attributed to inputs or sets
of inputs. This helps to get not just the individual parameter's sensitivities
but also gives a way to quantify the affect and sensitivity from
the interaction between the parameters.

```math
 Y = f_0+ \sum_{i=1}^d f_i(X_i)+ \sum_{i < j}^d f_{ij}(X_i,X_j) ... + f_{1,2...d}(X_1,X_2,..X_d)
```

```math
 Var(Y) = \sum_{i=1}^d V_i + \sum_{i < j}^d V_{ij} + ... + V_{1,2...,d}
```

The Sobol Indices are "order"ed, the first order indices given by ``S_i = \frac{V_i}{Var(Y)}``
the contribution to the output variance of the main effect of `` X_i ``, therefore it
measures the effect of varying `` X_i `` alone, but averaged over variations
in other input parameters. It is standardised by the total variance to provide a fractional contribution.
Higher-order interaction indices `` S_{i,j}, S_{i,j,k} `` and so on can be formed
by dividing other terms in the variance decomposition by `` Var(Y) ``.

### API

```julia
function gsa(f, method::Sobol, A::AbstractMatrix{TA}, B::AbstractMatrix;
             batch=false, Ei_estimator = :Jansen1999, distributed::Val{SHARED_ARRAY} = Val(false), kwargs...) where {TA, SHARED_ARRAY}

```

`Ei_estimator` can take `:Homma1996`, `:Sobol2007` and `:Jansen1999` for which
  Monte Carlo estimator is used for the `Ei` term. Defaults to `:Jansen1999`. Details for these can be found in the 
  corresponding papers:
    - `:Homma1996` - [Homma, T. and Saltelli, A., 1996. Importance measures in global sensitivity analysis of nonlinear models. Reliability Engineering & System Safety, 52(1), pp.1-17.](https://www.sciencedirect.com/science/article/abs/pii/0951832096000026)
    - `:Sobol2007` - [I.M. Sobol, S. Tarantola, D. Gatelli, S.S. Kucherenko and W. Mauntz, 2007, Estimating the approx- imation errors when fixing unessential factors in global sensitivity analysis, Reliability Engineering and System Safety, 92, 957–960.](https://www.sciencedirect.com/science/article/abs/pii/S0951832006001499)
    [A. Saltelli, P. Annoni, I. Azzini, F. Campolongo, M. Ratto and S. Tarantola, 2010, Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index, Computer Physics Communications 181, 259–270.](https://www.sciencedirect.com/science/article/abs/pii/S0010465509003087)
    - `:Jansen1999` - [M.J.W. Jansen, 1999, Analysis of variance designs for model output, Computer Physics Communi- cation, 117, 35–43.](https://www.sciencedirect.com/science/article/abs/pii/S0010465598001544)


### Example

```julia
using GlobalSensitivity, QuasiMonteCarlo

function ishi(X)
    A= 7
    B= 0.1
    sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1])
end

n = 600000
lb = -ones(4)*π
ub = ones(4)*π
sampler = SobolSample()
A,B = QuasiMonteCarlo.generate_design_matrices(n,lb,ub,sampler)

res1 = gsa(ishi,Sobol(order=[0,1,2]),A,B)

function ishi_batch(X)
    A= 7
    B= 0.1
    @. sin(X[1,:]) + A*sin(X[2,:])^2+ B*X[3,:]^4 *sin(X[1,:])
end

res2 = gsa(ishi_batch,Sobol(),A,B,batch=true)
```
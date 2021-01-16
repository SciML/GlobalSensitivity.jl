# Morris Method

```julia
struct Morris <: GSAMethod
    p_steps::Array{Int,1}
    relative_scale::Bool
    num_trajectory::Int
    total_num_trajectory::Int
    len_design_mat::Int
end
```

`Morris` has the following keyword arguments:

- `p_steps` - Vector of ``\Delta`` for the step sizes in each direction. Required.
- `relative_scale` - The elementary effects are calculated with the assumption that
  the parameters lie in the range `[0,1]` but as this is not always the case
  scaling is used to get more informative, scaled effects. Defaults to `false`.
- `total_num_trajectory`, `num_trajectory` - The total number of design matrices that are
  generated out of which `num_trajectory` matrices with the highest spread are used in calculation.
- `len_design_mat` - The size of a design matrix.

## Morris Method Details

The Morris method also known as Morris’s OAT method where OAT stands for
One At a Time can be described in the following steps:

We calculate local sensitivity measures known as “elementary effects”,
which are calculated by measuring the perturbation in the output of the
model on changing one parameter.

``EE_i = \frac{f(x_1,x_2,..x_i+ \Delta,..x_k) - y}{\Delta}``

These are evaluated at various points in the input chosen such that a wide
“spread” of the parameter space is explored and considered in the analysis,
to provide an approximate global importance measure. The mean and variance of
these elementary effects is computed. A high value of the mean implies that
a parameter is important, a high variance implies that its effects are
non-linear or the result of interactions with other inputs. This method
does not evaluate separately the contribution from the
interaction and the contribution of the parameters individually and gives the
effects for each parameter which takes into consideration all the interactions and its
individual contribution.

### API

`function gsa(f, method::Morris, p_range::AbstractVector; batch=false, kwargs...)`

### Example

Morris method on Ishigami function 

```julia
using GlobalSensitivity

function ishi(X)
    A= 7
    B= 0.1
    sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1])
end

lb = -ones(4)*π
ub = ones(4)*π

m = gsa(ishi, Morris(num_trajectory=500000), [[lb[i],ub[i]] for i in 1:4])
```
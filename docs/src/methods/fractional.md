# Fractional Factorial Method

`FractionalFactorial` does not have any keyword arguments.

## Method Details

Fractional Factorial method creates a design matrix by utilising
Hadamard Matrix and uses it run simulations of the input model.
The main effects are then evaluated by dot product between the contrast
for the parameter and the vector of simulation results. The
corresponding main effects and variance, i.e. square of the main effects
are returned as results for Fractional Factorial method.

### API

```@docs
gsa(f, method::FractionalFactorial; num_params, p_range = nothing, kwargs...)
```

### Example

```julia
using GlobalSensitivity, Test

f = X -> X[1] + 2 * X[2] + 3 * X[3] + 4 * X[7] * X[12]
res1 = gsa(f,FractionalFactorial(),num_params = 12,samples=10)
```

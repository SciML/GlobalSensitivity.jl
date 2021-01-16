# Global Sensitivity Analysis

Global Sensitivity Analysis (GSA) methods are used to quantify the uncertainty in
output of a model w.r.t. the parameters. These methods allow practitioners to 
measure both parameter's individual contributions and the contribution of their interactions
to the output uncertainity. 

## Installation

To use this functionality, you must install GlobalSensitivity.jl:

```julia
]add GlobalSensitivity
using GlobalSensitivity
```

Note: GlobalSensitivity.jl is unrelated to the GlobalSensitivityAnalysis.jl package.

## General Interface

The general interface for calling a global sensitivity analysis is either:

```julia
effects = gsa(f, method, param_range; N, batch=false)
```

where:

- `y=f(x)` is a function that takes in a single vector and spits out a single vector or scalar.
  If `batch=true`, then `f` takes in a matrix where each row is a set of parameters,
  and returns a matrix where each row is a the output for the corresponding row of parameters.
- `method` is one of the GSA methods below.
- `param_range` is a vector of tuples for the upper and lower bound for the given parameter `i`.
- `N` is a required keyword argument for the number of samples to take in the trajectories/design.

Note that for some methods there is a second interface where one can directly pass the design matrices:

```julia
effects = gsa(f, method, A, B; batch=false)
```

where `A` and `B` are design matrices with each row being a set of parameters. Note that `generate_design_matrices`
from [QuasiMonteCarlo.jl](https://github.com/JuliaDiffEq/QuasiMonteCarlo.jl) can be used to generate the design
matrices.

The descriptions of the available methods can be found in the Methods section.
The GSA interface allows for utilizing batched functions with the `batch` kwarg discussed above for parallel 
computation of GSA results.
# Global Sensitivity Analysis

Global Sensitivity Analysis (GSA) methods are used to quantify the uncertainty in
output of a model with respect to the parameters. These methods allow practitioners to
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

The general interface for performing global sensitivity analysis using this package is:

```@docs
gsa(f, method::GlobalSensitivity.GSAMethod, param_range; samples, batch = false)
```

The descriptions of the available methods can be found in the Methods section.
The `gsa` interface allows for utilizing batched functions with the `batch` kwarg discussed above for parallel
computation of GSA results.

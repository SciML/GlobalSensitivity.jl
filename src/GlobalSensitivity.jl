"""
    GlobalSensitivity

Global sensitivity analysis methods for quantifying how model inputs contribute to
output uncertainty.

Use [`gsa`](@ref) with method objects such as [`Sobol`](@ref), [`Morris`](@ref),
or [`RegressionGSA`](@ref) to estimate sensitivity indices from model evaluations
or precomputed design matrices.
"""
module GlobalSensitivity

import AbstractFFTs: rfft
import Combinatorics: permutations
import Copulas
import Copulas: GaussianCopula, IndependentCopula, SklarDist, condition
import Distributions: Distribution, MvNormal, Normal, Uniform, UnivariateDistribution, cdf, pdf
import FFTW: dct
import ForwardDiff
import KernelDensity: kde
import LinearAlgebra: Symmetric, diag, dot, pinv
import QuasiMonteCarlo
import Random
import Random: AbstractRNG, rand!, randperm, shuffle!
import RecursiveArrayTools
import Statistics: cov, mean, quantile, std, var
import StatsBase: competerank
import Trapz: trapz
using ComplexityMeasures: entropy, ValueHistogram, StateSpaceSet

abstract type GSAMethod end

include("morris_sensitivity.jl")
include("sobol_sensitivity.jl")
include("regression_sensitivity.jl")
include("DGSM_sensitivity.jl")
include("eFAST_sensitivity.jl")
include("delta_sensitivity.jl")
include("easi_sensitivity.jl")
include("rbd-fast_sensitivity.jl")
include("fractional_factorial_sensitivity.jl")
include("shapley_sensitivity.jl")
include("rsa_sensitivity.jl")
include("mutual_information_sensitivity.jl")

"""
    gsa(f, method::GSAMethod, param_range; samples, batch=false)

Perform global sensitivity analysis for a model or precomputed design matrices.

# Arguments

  - `f`: model function. It accepts one parameter vector and returns a scalar or vector.
  - `method`: global sensitivity analysis method.
  - `param_range`: lower and upper bounds for each parameter.

# Keyword Arguments

  - `samples`: required number of design-matrix samples, except for the fractional
    factorial and Morris methods.
  - `batch`: when `true`, `f` accepts parameter sets as matrix rows and returns one
    output row per input row.

# Alternative Interfaces

Additionally,

For [Delta Moment-Independent Method](@ref), [EASI Method](@ref) and [Regression Method](@ref) input and output matrix-based method as follows is available:

```julia
res = gsa(X, Y, method)
```

where:

- `X` is the number of parameters * samples matrix with parameter values.
- `Y` is the output dimension * number of samples matrix which are evaluated at `X`'s columns.
- `method` is one of the GSA methods below.

For [Sobol Method](@ref), one can use the following design matrices-based method instead of parameter range-based method discussed earlier:

```julia
effects = gsa(f, method, A, B; batch=false)
```

where `A` and `B` are design matrices, with each row being a set of parameters. Note that `generate_design_matrices`
from [QuasiMonteCarlo.jl](https://docs.sciml.ai/QuasiMonteCarlo/stable/) can be used to generate the design
matrices.
"""
function gsa(f, method::GSAMethod, param_range; samples, batch = false) end

export gsa

export Sobol, Morris, RegressionGSA, DGSM, eFAST, DeltaMoment, EASI, FractionalFactorial,
    RBDFAST, Shapley, RSA, MutualInformation

include("precompilation.jl")

end # module

module GlobalSensitivityAnalysis

using Statistics, RecursiveArrayTools, LinearAlgebra
using QuasiMonteCarlo, ForwardDiff
using Parameters: @unpack, @with_kw
using FFTW, Distributions

abstract type GSAMethod end

include("morris_sensitivity.jl")
include("sobol_sensitivity.jl")
include("regression_sensitivity.jl")
include("DGSM_sensitivity.jl")
include("eFAST_sensitivity.jl")


export Sobol, Morris, gsa,
       SensitivityAlg, RegressionGSA, DGSM, eFAST

end # module

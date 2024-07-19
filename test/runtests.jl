using SafeTestsets, Test

const GROUP = get(ENV, "GROUP", "All")

@time begin
    if GROUP == "All" || GROUP == "GSA"
        @time @safetestset "Morris Method" include("morris_method.jl")
        @time @safetestset "Sobol Method" include("sobol_method.jl")
        @time @safetestset "DGSM Method" include("DGSM.jl")
        @time @safetestset "eFAST Method" include("eFAST_method.jl")
        @time @safetestset "RegressionGSA Method" include("regression_sensitivity.jl")
        @time @safetestset "DeltaMoment Method" include("delta_method.jl")
        @time @safetestset "Fractional factorial method" include("fractional_factorial_method.jl")
        @time @safetestset "Rbd-fast method" include("rbd-fast_method.jl")
        @time @safetestset "Easi Method" include("easi_method.jl")
        @time @safetestset "Shapley Method" include("shapley_method.jl")
        @time @safetestset "KSRank Method" include("ks_rank_method.jl")
    end
end

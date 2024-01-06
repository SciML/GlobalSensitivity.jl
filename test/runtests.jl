using GlobalSensitivity, SafeTestsets
using Test

const GROUP = get(ENV, "GROUP", "All")

@time begin
    if GROUP == "All" || GROUP == "GSA"
        @time @safetestset "Quality Assurance" begin
            include("qa.jl")
        end
        @time @safetestset "Morris Method" begin
            include("morris_method.jl")
        end
        @time @safetestset "Sobol Method" begin
            include("sobol_method.jl")
        end
        @time @safetestset "DGSM Method" begin
            include("DGSM.jl")
        end
        @time @safetestset "eFAST Method" begin
            include("eFAST_method.jl")
        end
        @time @safetestset "RegressionGSA Method" begin
            include("regression_sensitivity.jl")
        end
        @time @safetestset "DeltaMoment Method" begin
            include("delta_method.jl")
        end
        @time @safetestset "Fractional factorial method" begin
            include("fractional_factorial_method.jl")
        end
        @time @safetestset "Rbd-fast method" begin
            include("rbd-fast_method.jl")
        end
        @time @safetestset "Easi Method" begin
            include("easi_method.jl")
        end
        @time @safetestset "Shapley Method" begin
            include("shapley_method.jl")
        end
    end
end

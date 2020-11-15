using GlobalSensitivity, SafeTestsets
using Test

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV,"APPVEYOR")
const is_TRAVIS = haskey(ENV,"TRAVIS")

@time begin

if GROUP == "All" || GROUP == "GSA"
    @time @safetestset "Morris Method" begin include("morris_method.jl") end
    @time @safetestset "Sobol Method" begin include("sobol_method.jl") end
    @time @safetestset "DGSM Method" begin include("DGSM.jl") end
    @time @safetestset "eFAST Method" begin include("eFAST_method.jl") end
    @time @safetestset "RegressionGSA Method" begin include("regression_sensitivity.jl") end
end

end

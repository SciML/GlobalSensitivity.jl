using GlobalSensitivity, Aqua, Test
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(GlobalSensitivity)
    Aqua.test_ambiguities(GlobalSensitivity, recursive = false)
    Aqua.test_deps_compat(GlobalSensitivity, check_extras = false)
    @test_broken false  # Aqua deps_compat: missing [compat] entry for Pkg extra — see https://github.com/SciML/GlobalSensitivity.jl/issues/239
    Aqua.test_piracies(GlobalSensitivity)
    Aqua.test_project_extras(GlobalSensitivity)
    Aqua.test_stale_deps(GlobalSensitivity)
    Aqua.test_unbound_args(GlobalSensitivity)
    Aqua.test_undefined_exports(GlobalSensitivity)
end

using GlobalSensitivity, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(GlobalSensitivity)
    Aqua.test_ambiguities(GlobalSensitivity, recursive = false)
    Aqua.test_deps_compat(GlobalSensitivity)
    Aqua.test_piracies(GlobalSensitivity)
    Aqua.test_project_extras(GlobalSensitivity)
    Aqua.test_stale_deps(GlobalSensitivity)
    Aqua.test_unbound_args(GlobalSensitivity)
    Aqua.test_undefined_exports(GlobalSensitivity)
end

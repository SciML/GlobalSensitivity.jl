using SciMLTesting, GlobalSensitivity, Test
run_qa(
    GlobalSensitivity;
    explicit_imports = true,
    aqua_kwargs = (; ambiguities = (; recursive = false)),
    ei_kwargs = (;
        # OtherPkg-non-public names accessed qualified (e.g. `Random.default_rng`);
        # de-facto public, just not yet declared `public` upstream.
        all_qualified_accesses_are_public = (;
            ignore = (
                :default_rng,              # Random
                :generate_design_matrices, # QuasiMonteCarlo
                :sample,                   # QuasiMonteCarlo
                :gradient,                 # ForwardDiff
                :hessian,                  # ForwardDiff
            ),
        ),
    ),
    # Heavy `using Statistics/Distributions/...` implicit imports; making each
    # explicit is a large, mechanical refactor — tracked in
    # https://github.com/SciML/GlobalSensitivity.jl/issues/245
    ei_broken = (:no_implicit_imports,)
)

# Precompilation workload for GlobalSensitivity.jl
# This file precompiles common code paths to improve TTFX (Time To First X)

using PrecompileTools

@setup_workload begin
    # Minimal test function (Ishigami-like)
    function _precompile_f(X)
        sin(X[1]) + 7.0 * sin(X[2])^2 + 0.1 * X[3]^4 * sin(X[1])
    end

    # Small parameter range for precompilation (use Float64 explicitly)
    _precompile_lb = [-3.14159, -3.14159, -3.14159]
    _precompile_ub = [3.14159, 3.14159, 3.14159]
    _precompile_p_range = [(_precompile_lb[i], _precompile_ub[i]) for i in 1:3]

    @compile_workload begin
        # Precompile Sobol method (most common)
        # Use small sample size to keep precompilation time reasonable
        _A, _B = QuasiMonteCarlo.generate_design_matrices(
            64, _precompile_lb, _precompile_ub,
            QuasiMonteCarlo.SobolSample(), 2
        )
        _sobol_result = gsa(_precompile_f, Sobol(order = [0, 1]), _A, _B)

        # Precompile Morris method
        _morris_result = gsa(
            _precompile_f,
            Morris(num_trajectory = 4, total_num_trajectory = 8, p_steps = fill(10, 3)),
            _precompile_p_range
        )

        # Precompile RegressionGSA method (matrix-based API)
        _X_reg = QuasiMonteCarlo.sample(64, _precompile_lb, _precompile_ub, QuasiMonteCarlo.SobolSample())
        _Y_reg = reshape([_precompile_f(_X_reg[:, j]) for j in 1:64], 1, 64)
        _reg_result = gsa(_X_reg, _Y_reg, RegressionGSA())

        # Precompile eFAST method (requires more samples based on num_harmonics)
        _efast_result = gsa(
            _precompile_f, eFAST(num_harmonics = 4),
            _precompile_p_range; samples = 100
        )

        # Precompile EASI method (matrix-based API)
        _Y_easi = [_precompile_f(_X_reg[:, j]) for j in 1:64]
        _easi_result = gsa(_X_reg, _Y_easi, EASI())
    end
end

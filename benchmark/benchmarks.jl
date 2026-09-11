using GlobalSensitivity, BenchmarkTools
using StableRNGs, Distributions

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

# Ishigami test function (standard GSA benchmark)
ishi(X) = sin(X[1]) + 7 * sin(X[2])^2 + 0.1 * X[3]^4 * sin(X[1])
p_range = fill([-π, π], 3)

# Morris linear test
A_m = [1.0 0.0; 3.0 0.0; 5.0 0.0; 7.0 0.0]
f_morris(x) = A_m * vec(x)

# =============================================================================
# Variance-based methods
# =============================================================================

SUITE["sobol"] = BenchmarkGroup()

SUITE["sobol"]["ishigami"] = @benchmarkable gsa(
    $ishi, Sobol(), $p_range; samples = 2000
)
SUITE["sobol"]["efast"] = @benchmarkable gsa(
    $ishi, eFAST(), $p_range; num_harmonics = 6, samples = 500
)

# =============================================================================
# Morris method
# =============================================================================

SUITE["morris"] = BenchmarkGroup()

SUITE["morris"]["linear"] = @benchmarkable gsa(
    $f_morris,
    Morris(p_steps = [10, 10], total_num_trajectory = 500, num_trajectory = 100),
    [[1, 5], [1, 5]]
)

# =============================================================================
# Regression / DGSM
# =============================================================================

SUITE["other"] = BenchmarkGroup()

SUITE["other"]["regression"] = @benchmarkable gsa(
    $ishi, RegressionGSA(), $p_range; samples = 2000
)
SUITE["other"]["dgsm"] = @benchmarkable gsa(
    $ishi, DGSM(), [Uniform(-π, π) for _ in 1:3]; samples = 500
)
SUITE["other"]["delta_moment"] = @benchmarkable gsa(
    $ishi, DeltaMoment(), $p_range; samples = 1000
)

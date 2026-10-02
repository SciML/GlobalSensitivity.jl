using GlobalSensitivity, QuasiMonteCarlo, Test, OrdinaryDiffEq
using Distributions: Normal, quantile

function ishi_batch(X)
    A = 7
    B = 0.1
    return @. sin(X[1, :]) + A * sin(X[2, :])^2 + B * X[3, :]^4 * sin(X[1, :])
end
function ishi(X)
    A = 7
    B = 0.1
    return sin(X[1]) + A * sin(X[2])^2 + B * X[3]^4 * sin(X[1])
end

function linear_batch(X)
    A = 7
    B = 0.1
    return @. A * X[1, :] + B * X[2, :]
end
function linear(X)
    A = 7
    B = 0.1
    return A * X[1] + B * X[2]
end

n = 600000
lb = -ones(4) * π
ub = ones(4) * π
sampler = SobolSample()
A, B = QuasiMonteCarlo.generate_design_matrices(n, lb, ub, sampler)

res1 = gsa(ishi, Sobol(order = [0, 1, 2]), A, B)
res2 = gsa(ishi_batch, Sobol(), A, B, batch = true)

@test res1.S1 ≈ [0.3139335358797363, 0.44235918402206326, 0.0, 0.0] atol = 1.0e-4
@test res2.S1 ≈ [0.3139335358797363, 0.44235918402206326, 0.0, 0.0] atol = 1.0e-4

@test res1.ST ≈ [0.5576009081644232, 0.44237102330046346, 0.24366241588532553, 0.0] atol = 1.0e-4
@test res2.ST ≈ [0.5576009081644232, 0.44237102330046346, 0.24366241588532553, 0.0] atol = 1.0e-4

@test res1.S2 ≈ [
    0.0 1.2954849537879917e-6 0.24368597775338205 6.64142747164482e-6;
    0.0 0.0 4.279200031668718e-5 1.2542212940962112e-5;
    0.0 0.0 0.0 -7.998213172266514e-7; 0.0 0.0 0.0 0.0
] atol = 1.0e-4

res1 = gsa(ishi, Sobol(order = [0, 1, 2], nboot = 20), A, B)
@test res1.S1 ≈ [0.3139335358797363, 0.44235918402206326, 0.0, 0.0] atol = 1.0e-4
@test res1.ST ≈ [0.5576009081644232, 0.44237102330046346, 0.24366241588532553, 0.0] atol = 1.0e-4
@test res1.S2 ≈ [
    0.0 1.2954849537879917e-6 0.24368597775338205 6.64142747164482e-6;
    0.0 0.0 4.279200031668718e-5 1.2542212940962112e-5;
    0.0 0.0 0.0 -7.998213172266514e-7; 0.0 0.0 0.0 0.0
] atol = 1.0e-4
@test res1.S1_Conf_Int ≈ [
    0.00025677429613976373,
    0.0002887134457830459,
    0.00014501412900342335,
    0.0,
] atol = 1.0e-6
@test res1.ST_Conf_Int ≈ [
    0.0001223019032979223,
    0.00025743925490551845,
    0.00018012100786659994,
    0.0,
] atol = 1.0e-6
@test res1.S2_Conf_Int ≈ [
    0.0 0.00039047613806956837 0.00034406519039858785 0.00040595660839326563;
    0.0 0.0 0.0002071949693901681 0.00020077471918021318;
    0.0 0.0 0.0 0.00017230794653437235; 0.0 0.0 0.0 0.0
] atol = 1.0e-6

# issue #181
@testset "CI" begin
    res80 = gsa(ishi, Sobol(order = [0, 1, 2], nboot = 20, conf_level = 0.8), A, B)
    res95 = gsa(ishi, Sobol(order = [0, 1, 2], nboot = 20, conf_level = 0.95), A, B)
    res99 = gsa(ishi, Sobol(order = [0, 1, 2], nboot = 20, conf_level = 0.99), A, B)

    # Point estimates do not depend on conf_level
    @test res80.S1 == res95.S1 == res99.S1
    @test res80.ST == res95.ST == res99.ST
    @test res80.S2 == res95.S2 == res99.S2

    # Confidence intervals depend on confidence levels
    @test res80.S1_Conf_Int != res95.S1_Conf_Int
    @test res80.ST_Conf_Int != res95.ST_Conf_Int
    @test res80.S2_Conf_Int != res95.S2_Conf_Int

    # Wider intervals for higher confidence levels
    @test all(res80.S1_Conf_Int .≤ res95.S1_Conf_Int .≤ res99.S1_Conf_Int)
    @test all(res80.ST_Conf_Int .≤ res95.ST_Conf_Int .≤ res99.ST_Conf_Int)
    @test all(res80.S2_Conf_Int .≤ res95.S2_Conf_Int .≤ res99.S2_Conf_Int)

    # Actually, the only thing changing is the z-multiplier,
    # so the normalized standard error estimates across runs must equal
    z80 = quantile(Normal(0.0, 1.0), (1 + 0.8) / 2)
    z95 = quantile(Normal(0.0, 1.0), (1 + 0.95) / 2)
    z99 = quantile(Normal(0.0, 1.0), (1 + 0.99) / 2)
    @test res80.S1_Conf_Int ./ z80 ≈ res95.S1_Conf_Int ./ z95
    @test res80.S1_Conf_Int ./ z80 ≈ res99.S1_Conf_Int ./ z99
    @test res80.ST_Conf_Int ./ z80 ≈ res95.ST_Conf_Int ./ z95
    @test res80.ST_Conf_Int ./ z80 ≈ res99.ST_Conf_Int ./ z99
    @test res80.S2_Conf_Int ./ z80 ≈ res95.S2_Conf_Int ./ z95
    @test res80.S2_Conf_Int ./ z80 ≈ res99.S2_Conf_Int ./ z99
end

res1 = gsa(linear, Sobol(), A, B)
res2 = gsa(linear_batch, Sobol(), A, B, batch = true)

@test res1.S1 ≈ [0.9997953478183109, 0.0002040399766938839, 0.0, 0.0] atol = 1.0e-4
@test res2.S1 ≈ [0.9997953478183109, 0.0002040399766938839, 0.0, 0.0] atol = 1.0e-4

@test res1.ST ≈ [0.9997953478183109, 0.0002040399766938839, 0.0, 0.0] atol = 1.0e-4
@test res2.ST ≈ [0.9997953478183109, 0.0002040399766938839, 0.0, 0.0] atol = 1.0e-4

@test_nowarn gsa(linear, Sobol(order = [0, 1, 2], nboot = 100), A, B)

#=
library(sensitivity)
ishigami.fun <- function(X) {
  A <- 7
  B <- 0.1
  sin(X[, 1]) + A * sin(X[, 2])^2 + B * X[, 3]^4 * sin(X[, 1])
}
n <- 600000
X1 <- data.frame(matrix(runif(4 * n,-pi,pi), nrow = n))
X2 <- data.frame(matrix(runif(4 * n,-pi,pi), nrow = n))
sobol2007(ishigami.fun, X1, X2)
sobolSalt(ishigami.fun, X1, X2, scheme="A")
sobolSalt(ishigami.fun, X1, X2, scheme="B")

library(sensitivity)
ishigami.fun <- function(X) {
  A <- 7
  B <- 0.1
  A * X[, 1] + B * X[, 2]
}
n <- 6000000
X1 <- data.frame(matrix(runif(4 * n,-pi,pi), nrow = n))
X2 <- data.frame(matrix(runif(4 * n,-pi,pi), nrow = n))
sobol2007(ishigami.fun, X1, X2)
sobolSalt(ishigami.fun, X1, X2, scheme="A")
sobolSalt(ishigami.fun, X1, X2, scheme="B")
=#

function ishi_linear(X)
    A = 7
    B = 0.1
    return [sin(X[1]) + A * sin(X[2])^2 + B * X[3]^4 * sin(X[1]), A * X[1] + B * X[2]]
end

function ishi_linear_batch(X)
    A = 7
    B = 0.1
    X1 = @. sin(X[1, :]) + A * sin(X[2, :])^2 + B * X[3, :]^4 * sin(X[1, :])
    X2 = @. A * X[1, :] + B * X[2, :]
    return vcat(X1', X2')
end

res1 = gsa(ishi_linear, Sobol(), A, B)
res2 = gsa(ishi_linear_batch, Sobol(), A, B, batch = true)

# Now both tests together

@test res1.S1[1, :] ≈ [0.3139335358797363, 0.44235918402206326, 0.0, 0.0] atol = 1.0e-4
@test res2.S1[1, :] ≈ [0.3139335358797363, 0.44235918402206326, 0.0, 0.0] atol = 1.0e-4

@test res1.ST[1, :] ≈ [0.5576009081644232, 0.44237102330046346, 0.24366241588532553, 0.0] atol = 1.0e-4
@test res2.ST[1, :] ≈ [0.5576009081644232, 0.44237102330046346, 0.24366241588532553, 0.0] atol = 1.0e-4

@test res1.S1[2, :] ≈ [0.9997953478183109, 0.0002040399766938839, 0.0, 0.0] atol = 1.0e-4
@test res2.S1[2, :] ≈ [0.9997953478183109, 0.0002040399766938839, 0.0, 0.0] atol = 1.0e-4

@test res1.ST[2, :] ≈ [0.9997953478183109, 0.0002040399766938839, 0.0, 0.0] atol = 1.0e-4
@test res2.ST[2, :] ≈ [0.9997953478183109, 0.0002040399766938839, 0.0, 0.0] atol = 1.0e-4

function f(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2] #prey
    return du[2] = -p[3] * u[2] + p[4] * u[1] * u[2] #predator
end

u0 = [1.0; 1.0]
tspan = (0.0, 10.0)
p = [1.5, 1.0, 3.0, 1.0]
prob = ODEProblem(f, u0, tspan, p)
t = collect(range(0, stop = 10, length = 200))

f1 = let prob = prob, t = t
    function (p)
        prob1 = remake(prob; p = p)
        sol = solve(prob1, Tsit5(); saveat = t)
        return sol
    end
end

m = gsa(f1, Sobol(), [[1, 5], [1, 5], [1, 5], [1, 5]], samples = 100)
@test m isa GlobalSensitivity.SobolResult
m = gsa(
    f1, Sobol(order = [0, 1, 2], nboot = 10), [[1, 5], [1, 5], [1, 5], [1, 5]],
    samples = 100
)
@test m isa GlobalSensitivity.SobolResult

@testset "Sobol non-finite filtering and extended result" begin
    # Backwards-compatible constructors
    res_6arg = GlobalSensitivity.SobolResult([0.5], nothing, nothing, nothing, [0.5], nothing)
    # Public type keeps four parameters (non-breaking for SobolResult{T1, T2, T3, T4} spellings)
    @test res_6arg isa GlobalSensitivity.SobolResult{Vector{Float64}, Nothing, Nothing, Nothing}
    @test res_6arg.VY === nothing
    @test res_6arg.n === nothing
    @test Sobol([0, 1], 1, 0.95).nonfinite === :propagate
    @test Sobol([0, 1], 1, 0.95, :drop).nonfinite === :drop
    @test_throws ArgumentError Sobol(nonfinite = :invalid)
    @test_throws ArgumentError Sobol([0, 1], 1, 0.95, :invalid)

    n_nan = 4000
    A_nan, B_nan = QuasiMonteCarlo.generate_design_matrices(n_nan, lb, ub, sampler)
    # Combination-shaped failure region, so the mask must account for the hybrid points
    fails(x) = x[1] + x[2] > 4.0
    ishi_nan(x) = fails(x) ? NaN : ishi(x)
    ishi_nan_batch(X) = [ishi_nan(X[:, k]) for k in 1:size(X, 2)]
    ishi_linear_nan(x) = fails(x) ? [NaN, NaN] : ishi_linear(x)

    # Rows whose A, B, A_B^i (and B_A^i for second order) points are all finite
    function finite_rows(A, B, second_order)
        d, n = size(A)
        return map(1:n) do k
            a, b = A[:, k], B[:, k]
            ok = !fails(a) && !fails(b)
            for i in 1:d
                a_mix = copy(a)
                a_mix[i] = b[i]
                ok &= !fails(a_mix)
                if second_order
                    b_mix = copy(b)
                    b_mix[i] = a[i]
                    ok &= !fails(b_mix)
                end
            end
            ok
        end
    end

    # Default (:propagate) keeps NaN in the result
    res_prop = gsa(ishi_nan, Sobol(), A_nan, B_nan)
    @test isnan(res_prop.VY)
    @test all(isnan, res_prop.S1)
    @test res_prop.n == n_nan

    # :drop on finite output is a no-op: identical indices, no warning
    res_clean = gsa(ishi, Sobol(), A_nan, B_nan)
    res_clean_drop = @test_logs gsa(ishi, Sobol(nonfinite = :drop), A_nan, B_nan)
    @test res_clean_drop.S1 == res_clean.S1
    @test res_clean_drop.ST == res_clean.ST
    @test res_clean_drop.n == n_nan

    # :drop equals :propagate on a design with the failing rows removed up front,
    # for every estimator and for first and second order
    for order in ([0, 1], [0, 1, 2]), Ei_estimator in (:Jansen1999, :Sobol2007, :Janon2014, :Homma1996)
        keep = finite_rows(A_nan, B_nan, 2 in order)
        @test 0 < count(keep) < n_nan
        ref = gsa(ishi, Sobol(order = order), A_nan[:, keep], B_nan[:, keep]; Ei_estimator)
        res = @test_logs (:warn, Regex("dropped $(n_nan - count(keep)) of $n_nan design rows")) gsa(
            ishi_nan, Sobol(order = order, nonfinite = :drop), A_nan, B_nan; Ei_estimator
        )
        @test res.n == count(keep)
        @test res.S1 ≈ ref.S1 rtol = 1.0e-12
        @test res.ST ≈ ref.ST rtol = 1.0e-12
        @test res.VY ≈ ref.VY rtol = 1.0e-12
        if 2 in order
            @test res.S2 ≈ ref.S2 rtol = 1.0e-12
        end
    end

    # Batch and non-batch agree
    res_drop = @test_logs (:warn, r"dropped") gsa(ishi_nan, Sobol(nonfinite = :drop), A_nan, B_nan)
    res_drop_batch = @test_logs (:warn, r"dropped") gsa(
        ishi_nan_batch, Sobol(nonfinite = :drop), A_nan, B_nan; batch = true
    )
    @test res_drop_batch.n == res_drop.n
    @test res_drop_batch.S1 ≈ res_drop.S1
    @test res_drop_batch.ST ≈ res_drop.ST

    # Multioutput: a row is dropped for all outputs; matches the prefiltered design
    keep = finite_rows(A_nan, B_nan, false)
    ref_multi = gsa(ishi_linear, Sobol(), A_nan[:, keep], B_nan[:, keep])
    res_multi = @test_logs (:warn, r"dropped") gsa(
        ishi_linear_nan, Sobol(nonfinite = :drop), A_nan, B_nan
    )
    @test res_multi.n == count(keep)
    @test res_multi.S1 ≈ ref_multi.S1 rtol = 1.0e-12
    @test res_multi.ST ≈ ref_multi.ST rtol = 1.0e-12

    # Per-replicate filtering with bootstrap: each replicate uses its own column chunk
    nboot = 4
    m = n_nan ÷ nboot
    expected_n = [
        count(finite_rows(A_nan[:, ((b - 1) * m + 1):(b * m)], B_nan[:, ((b - 1) * m + 1):(b * m)], true))
            for b in 1:nboot
    ]
    res_s2_boot = @test_logs (:warn, r"dropped") gsa(
        ishi_nan, Sobol(order = [0, 1, 2], nboot = nboot, nonfinite = :drop), A_nan, B_nan
    )
    @test res_s2_boot.n == expected_n
    @test all(isfinite, res_s2_boot.S1)
    @test all(isfinite, res_s2_boot.ST)
    @test all(isfinite, res_s2_boot.S2)
    @test all(isfinite, res_s2_boot.S1_Conf_Int)
    @test all(isfinite, res_s2_boot.ST_Conf_Int)
    @test all(isfinite, res_s2_boot.S2_Conf_Int)

    res_multi_boot = @test_logs (:warn, r"dropped") gsa(
        ishi_linear_nan, Sobol(nboot = nboot, nonfinite = :drop), A_nan, B_nan
    )
    @test res_multi_boot.n isa Vector{Int}
    @test all(isfinite, res_multi_boot.S1)
    @test all(isfinite, res_multi_boot.S1_Conf_Int)

    # Fewer than two kept rows is an error
    @test_throws ErrorException gsa(X -> NaN, Sobol(nonfinite = :drop), A_nan, B_nan)
    A_small = [0.1 0.9 0.9; 0.0 0.0 0.0]
    B_small = [0.2 0.9 0.9; 0.0 0.0 0.0]
    one_row_ok(x) = x[1] > 0.5 ? NaN : x[1] + x[2]
    @test_throws ErrorException gsa(one_row_ok, Sobol(nonfinite = :drop), A_small, B_small)
end

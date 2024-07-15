using GlobalSensitivity, Test, QuasiMonteCarlo


function ishi_batch(X)
    A = 7
    B = 0.1
    @. sin(X[1, :]) + A * sin(X[2, :])^2 + B * X[3, :]^4 * sin(X[1, :])
end
function ishi(X)
    A = 7
    B = 0.1
    sin(X[1]) + A * sin(X[2])^2 + B * X[3]^4 * sin(X[1])
end

function linear_batch(X)
    A = 7
    B = 0.1
    @. A * X[1, :] + B * X[2, :]
end
function linear(X)
    A = 7
    B = 0.1
    A * X[1] + B * X[2]
end

lb = -ones(4) * π
ub = ones(4) * π

res1 = gsa(ishi, KSRank(n_dummy_parameters=50), [[lb[i], ub[i]] for i in 1:4], samples = 100_000)
res2 = gsa(ishi_batch, KSRank(), [[lb[i], ub[i]] for i in 1:4], samples = 100_000, batch = true)

@test res1.S ≈ [0.348, 0.167, 0.00388, 0.00671] atol=1e-2
@test res2.S ≈ [0.348, 0.167, 0.00388, 0.00671] atol=1e-2

res1 = gsa(linear, KSRank(), [[lb[i], ub[i]] for i in 1:4], samples = 100_000)
res2 = gsa(linear_batch, KSRank(), [[lb[i], ub[i]] for i in 1:4], batch = true,
    samples = 100_000)
@test res1.S ≈ [0.993, 0.00611, 0.00399, 0.00519] atol=1e-2
@test res2.S ≈ [0.993, 0.00611, 0.00399, 0.00519] atol=1e-2


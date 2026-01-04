using GlobalSensitivity, Test, QuasiMonteCarlo, Random

function ishi(X)
    A = 7
    B = 0.1
    return sin(X[1]) + A * sin(X[2])^2 + B * X[3]^4 * sin(X[1])
end

lb = -ones(4) * π
ub = ones(4) * π

@time m = gsa(ishi, DeltaMoment(), fill([lb[1], ub[1]], 3), samples = 1000)
@test m.deltas ≈ [0.191604, 0.253396, 0.148682] atol = 3.0e-2

samples = 1000
X = QuasiMonteCarlo.sample(samples, lb, ub, QuasiMonteCarlo.SobolSample())
@time Y = ishi.([@view X[:, i] for i in 1:samples])

m = gsa(X[1:3, :], Y, DeltaMoment())
@test m.deltas ≈ [0.191604, 0.253396, 0.148682] atol = 3.0e-2

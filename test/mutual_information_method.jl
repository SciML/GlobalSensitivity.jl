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

res1 = gsa(
    ishi, MutualInformation(order=[0,1,2]), [[lb[i], ub[i]] for i in 1:4], samples = 10_000)

res2 = gsa(
ishi_batch, MutualInformation(order=[0,1,2]), [[lb[i], ub[i]] for i in 1:4],
samples = 10_000, batch = true)

@test res1.S1 ≈ [0.1416, 0.1929, 0.1204, 0.0925] atol = 1e-3
@test [0.09, 0.09, 0.09, 0.09] <= res1.S1_Conf_Int[:,1] <= [0.1, 0.1, 0.1, 0.1]
@test res2.S1 ≈ [0.1416, 0.1929, 0.1204, 0.0925] atol = 1e-3
@test [0.09, 0.09, 0.09, 0.09] <= res2.S1_Conf_Int[:,1] <= [0.1, 0.1, 0.1, 0.1]

@test sortperm(res1.ST) == [4,3,1,2]
@test sortperm(res2.ST) == [4,3,1,2]

@test res1.S2 ≈ [
    0.0       0.576849  0.656412  0.681677
    0.576849  0.0       0.609111  0.615966
    0.656412  0.609111  0.0       0.661516
    0.681677  0.615966  0.661516  0.0
] atol = 1e-2

@test res2.S2 ≈ [
    0.0       0.576849  0.656412  0.681677
    0.576849  0.0       0.609111  0.615966
    0.656412  0.609111  0.0       0.661516
    0.681677  0.615966  0.661516  0.0
] atol = 1e-2

res1 = gsa(
    linear, MutualInformation(), [[lb[i], ub[i]] for i in 1:4], samples = 10_000)
res2 = gsa(
    linear_batch, MutualInformation(), [[lb[i], ub[i]] for i in 1:4], batch = true,
    samples = 10_000)

@test res1.S1 ≈ [0.8155, 0.08997, 0.09096, 0.09747] atol = 1e-3
@test res2.S1 ≈ [0.8155, 0.08997, 0.09096, 0.09747] atol = 1e-3


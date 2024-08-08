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

function linear_int(X)
    A = 7
    B = 5.0
    A * X[1] + B * X[2] * X[3]
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
    ishi, MutualInformation(), [[lb[i], ub[i]] for i in 1:4], samples = 10_000)

res2 = gsa(
    ishi_batch, MutualInformation(), [[lb[i], ub[i]] for i in 1:4], samples = 10_000, batch = true)

res_sobol = gsa(
    ishi, Sobol(order = [0, 1, 2]), [[lb[i], ub[i]] for i in 1:4], samples = 10_000)
print(res1.S)
print(res2.S)
@test res1.S≈[0.2439708737198949, 0.5391319252363749, 0.12221504399804894, 0.0] atol=1e-4
@test res2.S≈[0.2439694207933475, 0.5394854849633759, 0.12179972423772056, 0.0] atol=1e-4

@test sortperm(res1.S) == [4, 3, 1, 2]
@test sortperm(res2.S) == [4, 3, 1, 2]

@test sortperm(res1.S) == sortperm(abs.(res_sobol.S1))
@test sortperm(res2.S) == sortperm(abs.(res_sobol.S1))

res1 = gsa(
    linear, MutualInformation(), [[lb[i], ub[i]] for i in 1:4], samples = 10_000)
res2 = gsa(
    linear_batch, MutualInformation(), [[lb[i], ub[i]] for i in 1:4], batch = true,
    samples = 10_000)

@test res1.S≈[4.5853400740929775, 0.0, 0.0, 0.0] atol=1e-4
@test res2.S≈[4.585864831042779, 0.0, 0.0, 0.0] atol=1e-4

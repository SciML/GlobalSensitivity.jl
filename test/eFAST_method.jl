using GlobalSensitivity, QuasiMonteCarlo, Test, OrdinaryDiffEq, Distributions

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

res1 = gsa(ishi, eFAST(), [[lb[i], ub[i]] for i in 1:4], samples = 15000)
res2 = gsa(ishi_batch, eFAST(), [[lb[i], ub[i]] for i in 1:4], samples = 15000,
    batch = true)

@test res1.S1≈[0.307599 0.442412 3.0941e-25 3.42372e-28] atol=1e-4
@test res2.S1≈[0.307599 0.442412 3.0941e-25 3.42372e-28] atol=1e-4

@test res1.ST≈[0.556244 0.446861 0.239259 0.027099] atol=1e-4
@test res2.ST≈[0.556244 0.446861 0.239259 0.027099] atol=1e-4

res1 = gsa(ishi, eFAST(), [Uniform(lb[i], ub[i]) for i in 1:4], samples = 15000)
res2 = gsa(ishi_batch, eFAST(), [Uniform(lb[i], ub[i]) for i in 1:4], samples = 15000,
    batch = true)

@test res1.S1≈[0.307599 0.442412 3.0941e-25 3.42372e-28] atol=1e-4
@test res2.S1≈[0.307599 0.442412 3.0941e-25 3.42372e-28] atol=1e-4

@test res1.ST≈[0.556244 0.446861 0.239259 0.027099] atol=1e-4
@test res2.ST≈[0.556244 0.446861 0.239259 0.027099] atol=1e-4

res1 = gsa(ishi, eFAST(), [Normal() for i in 1:4], samples = 15000)
res2 = gsa(ishi_batch, eFAST(), [Normal() for i in 1:4], samples = 15000,
    batch = true)

@test res1.S1≈[0.10140099594185588 0.7556923800497227 1.5688549609448593e-6 3.0948866309361236e-7] atol=1e-1
@test res2.S1≈[0.09791555320248665 0.7489863459743518 1.2521202417363353e-7 1.8830074456723684e-6] atol=1e-1

@test res1.ST≈[0.1538586603409846 0.84687840567574 0.12089782535494331 0.101911083206915] atol=1e-1
@test res2.ST≈[0.15130306101221003 0.8455036299750917 0.12229080086326627 0.15148183125412495] atol=1e-1

res1 = gsa(linear, eFAST(), [[lb[i], ub[i]] for i in 1:4], samples = 15000)
res2 = gsa(linear_batch, eFAST(), [[lb[i], ub[i]] for i in 1:4], batch = true,
    samples = 15000)

@test res1.S1≈[0.997504 0.000203575 2.1599e-10 2.18296e-10] atol=1e-4
@test res2.S1≈[0.997504 0.000203575 2.1599e-10 2.18296e-10] atol=1e-4

@test res1.ST≈[0.999796 0.000204698 7.26874e-7 7.59996e-7] atol=1e-4
@test res2.ST≈[0.999796 0.000204698 7.26874e-7 7.59996e-7] atol=1e-4

function ishi_linear(X)
    A = 7
    B = 0.1
    [sin(X[1]) + A * sin(X[2])^2 + B * X[3]^4 * sin(X[1]), A * X[1] + B * X[2]]
end

function ishi_linear_batch(X)
    A = 7
    B = 0.1
    X1 = @. sin(X[1, :]) + A * sin(X[2, :])^2 + B * X[3, :]^4 * sin(X[1, :])
    X2 = @. A * X[1, :] + B * X[2, :]
    vcat(X1', X2')
end

res1 = gsa(ishi_linear, eFAST(), [[lb[i], ub[i]] for i in 1:4], samples = 15000)
res2 = gsa(ishi_linear_batch, eFAST(), [[lb[i], ub[i]] for i in 1:4], samples = 15000,
    batch = true)

# Now both tests together

@test res1.S1≈[0.307595 0.442411 7.75353e-26 2.3468e-28
               0.997498 0.000203571 3.18996e-35 4.19822e-35] atol=1e-4
@test res2.S1≈[0.307598 0.442411 5.27085e-26 3.50751e-29
               0.997498 0.000203571 1.08441e-34 9.90366e-35] atol=1e-4

@test res1.ST≈[0.556246 0.446861 0.239258 0.027104
               0.999796 0.00020404 6.36917e-8 6.34754e-8] atol=1e-4
@test res2.ST≈[0.556243 0.446861 0.239258 0.0271024
               0.999796 0.00020404 6.35579e-8 6.36016e-8] atol=1e-4

function f(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2] #prey
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2] #predator
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

m = gsa(f1, eFAST(), [[1, 5], [1, 5], [1, 5], [1, 5]], samples = 1000)

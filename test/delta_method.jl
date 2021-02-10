using GlobalSensitivity, Test, OrdinaryDiffEq

A = reshape([1,0,2,3],2,2)
function f_morris(p)
    A*p
end

function linear_batch(X)
    A= 7
    B= 0.1
    @. A*X[1,:]+B*X[2,:]
end

function neg_linear_batch(X)
    A= -7
    B= 0.1
    @. A*X[1,:]+B*X[2,:]
end

function ishi(X)
    A= 7
    B= 0.1
    sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1])
end

function ishi_linear(X)
    A= 7
    B= 0.1
    [sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1]),A*X[1]+B*X[2]]
end

function ishi_linear_batch(X)
    A= 7
    B= 0.1
    X1 = @. sin(X[1,:]) + A*sin(X[2,:])^2+ B*X[3,:]^4 *sin(X[1,:])
    X2 = @. A*X[1,:]+B*X[2,:]
    vcat(X1',X2')
end

lb = -ones(4)*π
ub = ones(4)*π

m = gsa(ishi,Delta(),fill([lb[1], ub[1]], 3), N=1000)
@test m.deltas ≈ [0.191604, 0.253396, 0.148682] atol=3e-2
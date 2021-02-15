using GlobalSensitivity, Test, OrdinaryDiffEq

function ishi(X)
    A= 7
    B= 0.1
    sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1])
end

lb = -ones(4)*π
ub = ones(4)*π

m = gsa(ishi,DeltaMoment(),fill([lb[1], ub[1]], 3), N=1000)
@test m.deltas ≈ [0.191604, 0.253396, 0.148682] atol=3e-2
using GlobalSensitivity, Test

function ishi_batch(X)
    A= 7
    B= 0.1
    @. sin(X[1,:]) + A*sin(X[2,:])^2+ B*X[3,:]^4 *sin(X[1,:])
end
function ishi(X)
    A= 7
    B= 0.1
    sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1])
end

function linear_batch(X)
    A= 7
    B= 0.1
    @. A*X[1,:]+B*X[2,:]
end
function linear(X)
    A= 7
    B= 0.1
    A*X[1]+B*X[2]
end

lb = -ones(4)*π
ub = ones(4)*π

res1 = gsa(ishi,EASI(),[[lb[i],ub[i]] for i in 1:4],N=15000)
res2 = gsa(ishi_batch,EASI(),[[lb[i],ub[i]] for i in 1:4],N=15000,batch=true)

res1efast = gsa(ishi,eFAST(),[[lb[i],ub[i]] for i in 1:4],n=15000)
res2efast = gsa(ishi_batch,eFAST(),[[lb[i],ub[i]] for i in 1:4],n=15000,batch=true)

@test res1.S1 ≈ res1efast.S1[1,:] atol = 3e-2
@test res2.S1 ≈ res2efast.S1[1,:] atol = 3e-2

res1 = gsa(linear,EASI(),[[lb[i],ub[i]] for i in 1:4], N=15000)
res2 = gsa(linear_batch,EASI(),[[lb[i],ub[i]] for i in 1:4],batch=true, N=15000)

res1efast = gsa(linear,eFAST(),[[lb[i],ub[i]] for i in 1:4])
res2efast = gsa(linear_batch,eFAST(),[[lb[i],ub[i]] for i in 1:4],batch=true)

@test res1.S1 ≈ res1efast.S1[1,:] atol = 3e-2
@test res2.S1 ≈ res2efast.S1[1,:] atol = 3e-2

# function ishi_linear(X)
#     A= 7
#     B= 0.1
#     [sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1]),A*X[1]+B*X[2]]
# end

# function ishi_linear_batch(X)
#     A= 7
#     B= 0.1
#     X1 = @. sin(X[1,:]) + A*sin(X[2,:])^2+ B*X[3,:]^4 *sin(X[1,:])
#     X2 = @. A*X[1,:]+B*X[2,:]
#     vcat(X1',X2')
# end

# res1 = gsa(ishi_linear,EASI(),[[lb[i],ub[i]] for i in 1:4],N=15000)
# res2 = gsa(ishi_linear_batch,EASI(),[[lb[i],ub[i]] for i in 1:4],N=15000,batch=true)

# res1 = gsa(ishi_linear,eFAST(),[[lb[i],ub[i]] for i in 1:4],n=15000)
# res2 = gsa(ishi_linear_batch,eFAST(),[[lb[i],ub[i]] for i in 1:4],n=15000,batch=true)

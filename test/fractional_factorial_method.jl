using GlobalSensitivity, Test

f = X -> X[1] + 2 * X[2] + 3 * X[3] + 4 * X[7] * X[12]
res1 = gsa(f,FractionalFactorial(),num_params = 12,N=10)

@test res1[1][1:3] == [1.0, 2.0, 3.0]
@test all(res1[1][4:end] .== 0.0) 
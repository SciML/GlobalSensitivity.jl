using Copulas, Distributions, Combinatorics, LinearAlgebra, Random, GlobalSensitivity

Random.seed!(1234)

function ishi(X)
    A = 7
    B = 0.1
    sin(X[1]) + A * sin(X[2])^2 + B * X[3]^4 * sin(X[1])
end

function ishi_batch(X)
    A = 7
    B = 0.1
    @. sin(X[:, 1]) + A * sin(X[:, 2])^2 + B * X[:, 3]^4 * sin(X[:, 1])
end

function linear_batch(X)
    A = 7
    B = 0.1
    @. A * X[:, 1] + B * X[:, 2]
end
function linear(X)
    A = 7
    B = 0.1
    A * X[1] + B * X[2]
end

######################### Test ishi #########################

n_perms = -1;
n_var = 1_000;
n_outer = 100;
n_inner = 3;
n_boot = 60_000;
dim = 4;
margins = (Uniform(-pi, pi), Uniform(-pi, pi), Uniform(-pi, pi), Uniform(-pi, pi));
dependency_matrix = Matrix{Int}(I, dim, dim);
C = GaussianCopula(dependency_matrix);
input_distribution = SklarDist(C, margins);

method = Shapley(dim = dim,
    n_perms = n_perms,
    n_var = n_var,
    n_outer = n_outer,
    n_inner = n_inner,
    n_boot = n_boot);

#---> non batch
result = gsa(ishi, method, input_distribution, batch = false);

shapley_effects = []
for idx in range(1, dim)
    push!(shapley_effects, median(result.Shapley_indices[idx, :]))
end

@test shapley_effects[1]≈0.4541102166987666 atol=1e-4
@test shapley_effects[2]≈0.4291427404161051 atol=1e-4
@test shapley_effects[3]≈0.09943782238327539 atol=1e-4
@test shapley_effects[4]≈0.017521253643223787 atol=1e-4
#<---- non batch

#---> batch 
result = gsa(ishi_batch, method, input_distribution, batch = true);

shapley_effects = []
for idx in range(1, dim)
    push!(shapley_effects, median(result.Shapley_indices[idx, :]))
end

@test shapley_effects[1]≈0.4541102166987666 atol=1e-4
@test shapley_effects[2]≈0.4291427404161051 atol=1e-4
@test shapley_effects[3]≈0.09943782238327539 atol=1e-4
@test shapley_effects[4]≈0.017521253643223787 atol=1e-4
#<--- batch 

######################### Test ishi #########################

######################### Test linear #########################

n_perms = -1;
n_var = 1_000;
n_outer = 100;
n_inner = 3;
n_boot = 60_000;
dim = 2;
margins = (Uniform(-pi, pi), Uniform(-pi, pi));
dependency_matrix = Matrix{Int}(I, dim, dim);
C = GaussianCopula(dependency_matrix);
input_distribution = SklarDist(C, margins);

method = Shapley(dim = dim,
    n_perms = n_perms,
    n_var = n_var,
    n_outer = n_outer,
    n_inner = n_inner,
    n_boot = n_boot);

#---> non batch 
result = gsa(linear, method, input_distribution, batch = false);

shapley_effects = []
for idx in range(1, dim)
    push!(shapley_effects, median(result.Shapley_indices[idx, :]))
end

@test shapley_effects[1]≈0.9337925131867066 atol=1e-4
@test shapley_effects[2]≈0.06620748681329328 atol=1e-4
#<--- non batch 

#---> batch 
result = gsa(linear_batch, method, input_distribution, batch = true);

shapley_effects = []
for idx in range(1, dim)
    push!(shapley_effects, median(result.Shapley_indices[idx, :]))
end

@test shapley_effects[1]≈0.9337925131867066 atol=1e-4
@test shapley_effects[2]≈0.06620748681329328 atol=1e-4
#<--- batch

######################### Test linear #########################

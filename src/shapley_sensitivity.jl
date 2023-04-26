"""
A working implementation of shapley values tested on ishi function
Working on changing this into more traditional GSA file structure

Main to-do:
    - Get the code into GSA class scaffolding
    - Make this implementation high performance (multi thread for loops)
"""


using Copulas, Distributions, Combinatorics, LinearAlgebra

function sub_sampling(distribution, n_sample, idx)
    # get the margins of the input distirbution 
    margins_sub = [input_distribution.m[Int(j)] for j in idx];
    # get the original correlation matrix
    sigma = distribution.C.Σ;
    # get a subset of the correlation matrix to define the new copula 
    copula_sub = GaussianCopula(sigma[:, idx][idx, :]);
    # create the subset distribution 
    dist_sub = SklarDist(copula_sub,margins_sub);
    # sample from the subset distirbution 
    sample = copy(transpose(rand(dist_sub, n_sample)));

    return sample
end

function condMVN_new(cov, dependent_ind, given_ind, X_given)

    B = cov[:, dependent_ind];
    B = B[dependent_ind, :];

    C = cov[:, dependent_ind];
    C = C[given_ind, :];

    D = cov[:, given_ind];
    D = D[given_ind, :];

    CDinv = transpose(C) * inv(D);
    condMean = CDinv * X_given;
    condVar = B - CDinv * C;

    return condMean, condVar

end 

function cond_sampling_new(distribution, n_sample, idx, idx_c, x_cond)

    # select the correct marginal distributions 
    margins_dep = [distribution.m[Int(i)] for i in idx];
    margins_cond = [distribution.m[Int(i)] for i in idx_c];

    # create a conditioned variable that follows a normal distribution 
    u_cond = zeros(size(x_cond));
    for (i, marginal) in enumerate(margins_cond)
        u_cond[i] = quantile.(Normal(), cdf(marginal, x_cond[i]));
    end

    sigma = distribution.C.Σ;
    cond_mean, cond_var = condMVN_new(sigma, idx, idx_c, u_cond);
    
    n_dep = length(idx);

    if n_dep == 1
        dist_cond = Normal(cond_mean[1,1], cond_var[1,1]);
        sample_norm = rand(dist_cond, n_sample);
    else
        dist_cond = MvNormal(cond_mean, cond_var);
        sample_norm = rand(dist_cond, n_sample);
        sample_norm = copy(transpose(sample_norm));
    end
    
    sample_x = zeros((n_sample, n_dep));
    ϕ = x -> cdf(Normal(), x);
    for i in 1:n_dep
        #@TODO
        u_i = ϕ(sample_norm[:, i]);
        sample_x[:, i] = quantile.(margins_dep[i], u_i)
    end
    
    return sample_x

end 

# define the function you want to test 
function test_func_ishi(A:: Number, B:: Number, batch)
    if batch
        func(X::Matrix) = @. sin(X[:, 1]) + A * sin(X[:, 2])^2 + B * X[:, 3]^4 * sin(X[:, 1]);
    else
        func(X::Array) = sin(X[1]) + A * sin(X[2])^2 + B * X[3]^4 * sin(X[1]);
    end

    return func
end


#############################


# define the global variables 
n_perms = nothing; # The number of permutations. If None, the exact permutations method is considerd.
n_var = 1000; # The sample size for the output variance estimation.
n_outer = 100; # The number of conditionnal variance estimations.
n_inner = 3; # The sample size for the conditionnal output variance estimation.
dim = 3;
n_boot = 500;
n_realization=1;

batch = true
f = test_func_ishi(7, 0.3, batch);

# define the input distirbution 
margins = (Uniform(-pi, pi), Uniform(-pi, pi), Uniform(-pi, pi));
# dependency_matrix = [4 -1 0.1 ; -1 4 -1 ; 0.1  -1 4]; # features are correlated 
dependency_matrix = [1 0 0; 0 1 0; 0 0 1]; # features are independent 
C = GaussianCopula(dependency_matrix);
input_distribution = SklarDist(C,margins);

# build the sample to run the model on --> shapley values will be calculated on these outputs 
n_realization = 1 # The number of realization if the model is a random meta-model.
if isnothing(n_perms)
    estimation_method = "exact";
    perms = collect(permutations(range(1,dim), dim));
    n_perms = length(perms);
else
    estimation_method = "random";
    perms = collect(permutations(range(1,dim), n_perms));
end

# Creation of the design matrix
input_sample_1 = copy(transpose(rand(input_distribution, n_var))); # number of samples x number of features 
input_sample_2 = zeros((n_perms * (dim - 1) * n_outer * n_inner, dim));

#---> iterate to create the sample 
for (i_p, perm) in enumerate(perms)
    idx_perm_sorted = sortperm(perm) # Sort the variable ids
    # println("For perm $i_p, perm = $perm")
    for j in 1:(dim-1)
        # normal set
        idx_j = perm[1:j];
        # Complementary set
        idx_j_c = perm[j+1:end];
        sample_j_c = sub_sampling(input_distribution, n_outer, idx_j_c);
        # println("j = $j, idx_j $idx_j, idx_j_c $idx_j_c")
    
        for l in range(1,size(sample_j_c)[1])
            xjc = sample_j_c[l, :];
            # Sampling of the set conditionally to the complementary element 
            xj = cond_sampling_new(input_distribution, n_inner, idx_j, idx_j_c, xjc);
            xx = hcat(xj, repeat(transpose(xjc), n_inner));
            ind_inner = (i_p - 1) * (dim - 1) * n_outer * n_inner + (j-1) * n_outer * n_inner + (l-1) * n_inner; # subtract 1 from all indices
            ind_inner = ind_inner + 1
            input_sample_2[ind_inner:ind_inner + n_inner - 1, :] = xx[:, idx_perm_sorted]
        end 
    end
end

X = cat(input_sample_1, input_sample_2, dims=1); 

if n_realization == 1
    output_sample = f(X);
else
    #@TODO need to figure out what n_realization is doing 
    output_sample = f(X, n_realization)
end

output_sample_1 = output_sample[1:n_var]
output_sample_2 = reshape(output_sample[n_var+1:end], (n_perms, dim-1, n_outer, n_inner, n_realization))
self_output_sample_2 = output_sample_2
println("DONE BUILDING SAMPLE")
# <----

#---> compute indices now 
shapley_indices = zeros(dim, n_boot, n_realization);
shapley_indices_2 = zeros(dim, n_realization);
c_hat = zeros(n_perms, dim, n_boot, n_realization);

if estimation_method == "random"
    boot_perms = zeros(Int, n_perms, n_boot)
end

variance = zeros(n_boot, n_realization)

for i in range(1, n_boot)
    # Bootstrap sample indexes
    # The first iteration is computed over the all sample.
    if i > 1
        boot_var_idx = rand(1:n_var, n_var);
        if estimation_method == "exact"
            boot_No_idx = rand(1:n_outer, n_outer);
        else
            
            boot_n_perms_idx = rand(1:n_perms, n_perms)
            boot_perms[:, i] = boot_n_perms_idx
        end
    else
        boot_var_idx = range(1, n_var);
        if estimation_method == "exact"
            boot_No_idx = range(1, n_outer);
        else
            boot_n_perms_idx = range(1, n_perms);
            boot_perms[:, i] = boot_n_perms_idx;
        end
    end

    var_y = var(output_sample_1[boot_var_idx], corrected=false);
    variance[i] = var_y;

    # Conditional variances
    if estimation_method == "exact"
        output_sample_2 = self_output_sample_2[:, :, boot_No_idx, :, :];
    else
        # @TODO this will probably throw some index error, but fix when debugging the random mode 
        output_sample_2 = self_output_sample_2[boot_n_perms_idx]
    end

    c_var = var(output_sample_2, dims=4, corrected=false);
    c_var = dropdims(c_var; dims=4); # need to squeeze the dimension over which we applied the variance operator. Julia does not do it automatically
    
    c_mean_var = mean(c_var, dims=3);
    c_mean_var = dropdims(c_mean_var; dims=3); # need to squeeze the dimension over which we applied the mean operator. Julia does not do it automatically

    # Cost estimation
    c_hat[:, :, i] = cat(c_mean_var, repeat([var_y], n_perms), dims = 2);

end

# Cost variation
delta_c = copy(c_hat);
delta_c[:, 2:end, :, :] = c_hat[:, 2:end, :, :] - c_hat[:, 1:end-1, :, :]; # check on whether done correctly in julia

# Cost variation
for i in range(1, n_boot)

    if estimation_method == "random"
        boot_n_perms_idx = boot_perms[:, i];
        tmp_perms = perms[boot_n_perms_idx, :];
    else
        tmp_perms = perms;
    end

    # estimate shapley 
    for i_p in range(1, length(tmp_perms))
        perm = perms[i_p];
        shapley_indices[perm, i] += delta_c[i_p, :, i];
        shapley_indices_2[perm] += delta_c[i_p, :, 1].^2;
    end
end

output_variance = reshape(variance, (1, size(variance)[1], size(variance)[2]));
shapley_indices = shapley_indices ./ n_perms ./ output_variance;

if estimation_method == "random"
    output_variance_2 = output_variance[:, 1, :];
    shapley_indices_2 = shapley_indices_2 ./ n_perms ./ output_variance_2.^2;
    shapley_indices_SE = sqrt((shapley_indices_2 - shapley_indices[:, 0].^2) ./ n_perms);

else
    shapley_indices_SE = nothing;
    total_indices_SE = nothing;
    first_indices_SE = nothing;
end

println("DONE COMPUTING INDICES")
med1 = median(shapley_indices[1, :]);
med2 = median(shapley_indices[2, :]);
med3 = median(shapley_indices[3, :]);
println("median shapley index for feature 1 = $med1")
println("median shapley index for feature 2 = $med2")
println("median shapley index for feature 3 = $med3")
#<----




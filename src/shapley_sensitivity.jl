
using Copulas, Distributions, Combinatorics, LinearAlgebra

struct Shapley 
    n_boot::Int
    n_var::Int
    n_outer::Int
    n_inner::Int
    corr_matrix::Array
end

Shapley(; n_boot = 500, n_var = 1000, n_outer = 100, n_inner = 3, corr_matrix = [1 0 0; 0 1 0; 0 0 1]) = Shapley(n_boot, n_var, n_outer, n_inner, corr_matrix);

mutable struct ShapleyResult{T1,T2}
    Shapley_indices::T1
    output_variance::T2
end

function sub_sampling(distribution, n_sample, idx)

    # get the margins of the input distirbution 
    margins_sub = [distribution.m[Int(j)] for j in idx];
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
        u_i = ϕ(sample_norm[:, i]);
        sample_x[:, i] = quantile.(margins_dep[i], u_i)
    end
    
    return sample_x

end 

########### IMPLEMENTATION ##################################################################
function gsa(f::Any, method::Shapley, p_range::AbstractVector;batch=false)

    # Extract variables from shapley structure
    n_boot= method.n_boot
    n_var=method.n_var
    n_outer = method.n_outer
    n_inner= method.n_inner
    dim = length(p_range);
    dependency_matrix = method.corr_matrix;  
    perms = collect(permutations(range(1,dim), dim));
    n_perms = length(perms);
    
    # define the input distirbution 
    margins = [];
    for i in range(1, length(p_range))
        push!(margins, Uniform(p_range[i][1], p_range[i][2]))
    end
 
    C = GaussianCopula(dependency_matrix);
    input_distribution = SklarDist(C,margins);

    ####### GENERATE SAMPLES ######ß
    # Creation of the design matrix
    input_sample_1 = copy(transpose(rand(input_distribution, n_var))); # number of samples x number of features 
    input_sample_2 = zeros((n_perms * (dim - 1) * n_outer * n_inner, dim));

    #---> iterate to create the sample 
    for (i_p, perm) in enumerate(perms)
        idx_perm_sorted = sortperm(perm) # Sort the variable ids
        
        for j in 1:(dim-1)
            # normal set
            idx_j = perm[1:j];
            # Complementary set
            idx_j_c = perm[j+1:end];
            sample_j_c = sub_sampling(input_distribution, n_outer, idx_j_c);
           
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

    # when batch is false, you are getting [[], [], ...] fix this to [...]
    if batch
        output_sample = f(X);
    else
        f1 = X -> [f(X[j, :]) for j in axes(X, 1)];
        output_sample = f1(X);
        output_sample = vcat(output_sample...);
    end

    output_sample_1 = output_sample[1:n_var];
    n_realization = 1;
    output_sample_2 = reshape(output_sample[n_var+1:end], (n_perms, dim-1, n_outer, n_inner, n_realization));
    self_output_sample_2 = output_sample_2;
    # <----

    ###### CALCULATE INDICES #########
    shapley_indices = zeros(dim, n_boot, n_realization);
    shapley_indices_2 = zeros(dim, n_realization);
    c_hat = zeros(n_perms, dim, n_boot, n_realization);

    variance = zeros(n_boot, n_realization)

    for i in range(1, n_boot)
        # Bootstrap sample indexes
        # The first iteration is computed over the all sample.
        if i > 1
            boot_var_idx = rand(1:n_var, n_var);
            boot_No_idx = rand(1:n_outer, n_outer);
        else
            boot_var_idx = range(1, n_var);
            boot_No_idx = range(1, n_outer);
        end

        var_y = var(output_sample_1[boot_var_idx], corrected=false);
        variance[i] = var_y[1];

        # Conditional variances
        output_sample_2 = self_output_sample_2[:, :, boot_No_idx, :, :];

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

        # estimate shapley 
        for i_p in range(1, length(perms))
            perm = perms[i_p];
            shapley_indices[perm, i] += delta_c[i_p, :, i];
            shapley_indices_2[perm] += delta_c[i_p, :, 1].^2;
        end
    end

    output_variance = reshape(variance, (1, size(variance)[1], size(variance)[2]));
    shapley_indices = shapley_indices ./ n_perms ./ output_variance;

    ShapleyResult(shapley_indices, output_variance)
end


############################ TEST ######################################

function ishi_batch(X)
    A = 7
    B = 0.1
    @. sin(X[:, 1]) + A * sin(X[:, 2])^2 + B * X[:, 3]^4 * sin(X[:, 1])
end

function ishi(X)
    A = 7
    B = 0.1
    sin(X[1]) + A * sin(X[2])^2 + B * X[3]^4 * sin(X[1])
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

test_funcs = [ishi_batch, ishi, linear_batch, linear];
f = test_funcs[1];

# define shapley params
n_var = 1000; # The sample size for the output variance estimation.
n_outer = 100; # The number of conditionnal variance estimations.
n_inner = 3; # The sample size for the conditionnal output variance estimation.
n_boot = 500;

# function specific arguments
# you need to define this differently based on (1) if features are correlated (2) Number of inputs. Needs to be positive definite
corr_matrix = Matrix{Float64}(I, 3, 3); # ishi function 
# corr_matrix = Matrix{Float64}(I, 2, 2); # linear function 

# need to define based on the number of features 
p_range = [(-pi, pi), (-pi, pi), (-pi, pi)]; # ishi function
# p_range = [(-pi, pi), (-pi, pi)]; # linear function

# define the shapley method object
method = Shapley(n_boot=n_boot, n_inner=n_inner, n_outer=n_outer, n_var=n_var, corr_matrix=corr_matrix);

# compute
output = gsa(f, method, p_range, batch=true);
output
@doc raw"""

    Shapley(dim::Int; nboot::Int=500, n_perms::Int=-1,n_var::Int=1000, n_outer::Int=100, n_inner::Int=3) 

- `dim`: number of features 
- `nboot`: for confidence interval calculation `nboot` should be specified for the number (>0) of bootstrap runs.
- `n_perms`: The number of permutations. If None, the exact permutations method is considerd.
- `n_var`: The sample size for the output variance estimation.
- `n_outer`: The number of conditionnal variance estimations.
- `n_inner`: The sample size for the conditionnal output variance estimation.


## Method Details

Shapley values come from cooperative game theory and were introduced 1953. 
They help determine the contribution of each player in the total payoff achieved by the coalition of the players. 
The sum of Shapley effects over all individual variable is equal to the variance; this normalization property gives us a better interpretability in 
determining the relative importance of variables [1, 2]. This also makes Shapley values more fair as they do not assign large importances to few inputs.
Sobol indices assume that all the inputs are independent, which may be erroneous in many real-world scenarios like 
physiology based pharmacokinetic models of the organs of the human body. 
Shapley indices consider that parameters of a model can be correlated. 

[1]:Owen, A. B., & Prieur, C. (2017). On Shapley value for measuring importance of dependent inputs. SIAM/ASA Journal on Uncertainty Quantification, 5(1), 986-1002.
[2]:Song, E., Nelson, B. L., & Staum, J. (2016). Shapley effects for global sensitivity analysis: Theory and computation. SIAM/ASA Journal on Uncertainty Quantification, 4(1), 1060-1083.


## API

    gsa(f, method::Shapley, input_distribution::SklarDist; kwargs...)

`input_distribution` is the joint distribution of the feature vectors.
To create this, first define the marginal distirbutions (uniform distirbutions). Then define their 
correlation using a copula (Copulas.jl can be used), Combine marginals and copula into a joint distirbution.
### Example

```julia
using GlobalSensitivity, Copulas

function ishi(X)
    A= 7
    B= 0.1
    sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1])
end

function ishi_batch(X)
    A= 7
    B= 0.1
    @. sin(X[:,1]) + A*sin(X[:,2])^2+ B*X[:,3]^4 *sin(X[:,1])
end

margins = (Uniform(-pi, pi), Uniform(-pi, pi), Uniform(-pi, pi), Uniform(-pi, pi));
dependency_matrix = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
dim = 4;
C = GaussianCopula(dependency_matrix);
input_distribution = SklarDist(C,margins);
method = Shapley(dim);

res1 = gsa(ishi,method,input_distribution,batch=false)

res2 = gsa(ishi,method,input_distribution,batch=true)
```
"""

## Data structures
struct Shapley <: GSAMethod
    nboot::Int
    n_perms::Int
    n_var::Int
    n_outer::Int
    n_inner::Int
    dim::Int
end

mutable struct ShapleyResult{T1,T2}
    Shapley_indices::T1
    output_variance::T2
end

Shapley(dim::Int; nboot::Int=500, n_perms::Int=-1,n_var::Int=1000, n_outer::Int=100, n_inner::Int=3) = Shapley(dim; nboot, n_perms, n_var, n_outer, n_inner)

################# HELPER FUNCTIONS FOR SAMPLING ##############
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
##############################################################################



########### IMPLEMENTATION ####################################################
function gsa(f, method::Shapley, input_distribution::SklarDist;batch=false)

    # Extract variables from shapley structure
    n_perms = method.n_perms
    nboot = method.nboot
    n_var = method.n_var
    n_outer = method.n_outer
    n_inner = method.n_inner
    dim = method.dim

    ### Generate sample ###
    if (n_perms==-1)
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

    if batch
        output_sample = f(X);
    else
        f1 = X -> [f(X[j, :]) for j in axes(X, 1)];
        output_sample = f1(X);
    end

    output_sample_1 = output_sample[1:n_var]
    output_sample_2 = permutedims(reshape(output_sample[n_var+1:end], (1,n_inner,n_outer,dim-1,n_perms)), (5,4,3,2,1))
    self_output_sample_2 = copy(output_sample_2)

    # <----

    #---> compute indices now 
    shapley_indices = zeros(dim, nboot, 1);
    shapley_indices_2 = zeros(dim, 1);
    c_hat = zeros(n_perms, dim, nboot, 1);

    if estimation_method == "random"
        boot_perms = zeros(Int, n_perms, nboot)
    end

    variance = zeros(nboot, 1)

    for i in range(1, nboot)
        # Bootstrap sample indexes
        # The first iteration is computed over the all sample.
        if i > 1
            boot_var_idx = rand(1:n_var, n_var);
            # boot_var_idx = range(1, n_var);
            if estimation_method == "exact"
                boot_No_idx = rand(1:n_outer, n_outer);
                # boot_No_idx = [20, 43, 1, 66, 97, 12, 15, 33, 23, 12, 5, 38, 1, 10, 6, 6, 38, 75, 85, 50, 51, 2, 76, 29, 9, 72, 5, 53, 65, 43, 79, 62, 76, 47, 76, 21, 18, 89, 13, 94, 92, 85, 6, 48, 40, 26, 30, 5, 15, 63, 37, 59, 95, 96, 80, 32, 62, 10, 12, 93, 64, 29, 90, 22, 19, 5, 36, 83, 5, 92, 76, 26, 83, 12, 7, 52, 80, 29, 1, 87, 79, 74, 3, 61, 67, 16, 73, 84, 56, 62, 56, 48, 17, 57, 9, 62, 79, 78, 33, 4]
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

        var_y = var(output_sample_1[boot_var_idx]);
        variance[i] = var_y;
        # println("For i = $i, var_y is = $var_y")

        # Conditional variances
        if estimation_method == "exact"
            output_sample_2 = self_output_sample_2[:, :, boot_No_idx, :, :];
        else
            # @TODO this will probably throw some index error, but fix when debugging the random mode 
            output_sample_2 = self_output_sample_2[boot_n_perms_idx]
        end

        c_var = var(output_sample_2, dims=4);
        c_var = dropdims(c_var; dims=4); # need to squeeze the dimension over which we applied the variance operator. Julia does not do it automatically
        # c_var is the same 

        c_mean_var = mean(c_var, dims=3);
        c_mean_var = dropdims(c_mean_var; dims=3); # need to squeeze the dimension over which we applied the mean operator. Julia does not do it automatically
        # println("For i = $i, c_mean_var is = $c_mean_var")

        # # Cost estimation
        c_hat[:, :, i] = cat(c_mean_var, repeat([var_y], n_perms), dims = 2);
        
    end

    # Cost variation
    delta_c = copy(c_hat);
    delta_c[:, 2:end, :, :] = c_hat[:, 2:end, :, :] - c_hat[:, 1:end-1, :, :]; # check on whether done correctly in julia

    # Cost variation
    for i in range(1, nboot)

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

    return ShapleyResult(shapley_indices, output_variance)
end
###################









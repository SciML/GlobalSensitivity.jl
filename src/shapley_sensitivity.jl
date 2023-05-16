@doc raw"""

    Shapley(n_boot, n_perms, n_var, n_outer, n_inner, dim)

- `n_perms`: number of permutations to consider
- `n_boot`: number of Bootstrap runs
- `n_var`: size of each bootsrapped sample
- `n_outer`: number of samples to be taken to estiamte conditional variance 
- `n_inner`: size of each n_outer sample taken
- `dim`: number of features in the function 

## Method Details

Shapely effects is a variance based method to assign attribution
to each feature based on how sentitive the function to the feature. 
Shapley effects take into account that features could be dependent, which 
is not possible in previous methods like Sobol indices. In our implementation, we use Copulas.jl to define the joint
input distirbution as a SklarDist. 

## API

    gsa_parallel(f, method::Shapley, input_distribution::SklarDist;batch=false)
    

### Example

```julia
using Copulas, Distributions
include("path/to/shapley_serial.jl")

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

n_perms = -1; # -1 indicates that we want to consider all permutations. One can also use n_perms > 0
n_var = 1000;
n_outer = 100;
n_inner = 3
n_boot = 60_000;

dim = 3;
margins = (Uniform(-pi, pi), Uniform(-pi, pi), Uniform(-pi, pi));
dependency_matrix = 1* Matrix(I, dim, dim)

C = GaussianCopula(dependency_matrix);
input_distribution = SklarDist(C,margins);

method = Shapley(dim=dim, n_perms=n_perms, n_var = n_var, n_outer = n_outer, n_inner = n_inner, n_boot=n_boot);

# non_batch
result_non_batch = gsa(ishi,method,input_distribution,batch=false)
shapley_indices = res1.Shapley_indices

for i in range(1, dim)
    println("Median Shapley effect for feature $i = ", median(shapley_indices[i, :]))
end

println("")

# batch
result_batch = gsa(ishi_batch,method,input_distribution,batch=true)
shapley_indices = res1.Shapley_indices

for i in range(1, dim)
    println("Median Shapley effect for feature $i = ", median(shapley_indices[i, :]))
end

```
"""


using Copulas, Distributions, Combinatorics, LinearAlgebra, Random

## Data structures
struct Shapley
    n_boot::Int
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

function Shapley(;dim::Int=3, n_boot::Int=500, n_perms::Int=-1,n_var::Int=1000, n_outer::Int=100, n_inner::Int=3)
    Shapley(n_boot, n_perms, n_var, n_outer, n_inner, dim)
end

################# HELPER FUNCTIONS FOR SAMPLING ##############
function sample_subset(distribution::SklarDist, n_sample::Int, idx::Vector{Int})
    """
    Generate a subset of a joint distribution by selecting the given marginals and correlations. Sample n_sample from this subset distribution.
    """

    # get the margins of the input distirbution 
    margins_of_subset = [input_distribution.m[Int(j)] for j in idx];

    # get the original correlation matrix
    sigma = distribution.C.Σ;

    # get a subset of the correlation matrix to define the new copula 
    copula_subset = GaussianCopula(sigma[:, idx][idx, :]);

    # create the subset distribution 
    dist_subset = SklarDist(copula_subset, margins_of_subset);

    # sample from the subset distirbution 
    sample_from_subset = copy(transpose(rand(dist_subset, n_sample)));

    return sample_from_subset
end

function find_cond_mean_var(cov::Matrix, dependent_ind::Vector{Int}, given_ind::Vector{Int}, X_given::Vector{Float64})
    """
    Find the conditional mean and variance of the given distirbution 
    """

    mat_b = cov[:, dependent_ind];
    mat_b = mat_b[dependent_ind, :];

    mat_c = cov[:, dependent_ind];
    mat_c = mat_c[given_ind, :];

    mat_d = cov[:, given_ind];
    mat_d = mat_d[given_ind, :];

    mat_cdinv = transpose(mat_c) * inv(mat_d);
    conditional_mean = mat_cdinv * X_given;
    contional_var = mat_b - CDinmat_cdinvv * mat_c;

    return conditional_mean, contional_var

end 

function cond_sampling(distribution::SklarDist, n_sample::Int, idx::Vector{Int}, idx_c::Vector{Int}, x_cond::Vector{Float64})

    # select the correct marginal distributions for the two subsets of features
    margins_dependent = [distribution.m[Int(i)] for i in idx];
    margins_conditional = [distribution.m[Int(i)] for i in idx_c];

    # create a conditioned variable that follows a normal distribution 
    cond_norm_var = zeros(size(x_cond));
    for (i, marginal) in collect(enumerate(margins_conditional))
        cond_norm_var[i] = quantile.(Normal(), cdf(marginal, x_cond[i]));
    end

    corr_mat = distribution.C.Σ;
    cond_mean, cond_var = find_cond_mean_var(corr_mat, idx, idx_c, cond_norm_var);
    
    n_dep = length(idx);

    # need to sample from univariate normal and multivariate normal using different functions
    if n_dep == 1
        dist_cond = Normal(cond_mean[1,1], cond_var[1,1]);
        sample_norm = rand(dist_cond, n_sample);
    else
        dist_cond = MvNormal(cond_mean, cond_var);
        sample_norm = transpose(rand(dist_cond, n_sample));
    end
    
    final_sample = zeros((n_sample, n_dep));
    ϕ = x -> cdf(Normal(), x);
    for i in 1:n_dep
        final_sample[:, i] = quantile.(margins_dependent[i],  ϕ(sample_norm[:, i]))
    end
    
    return final_sample

end 
##############################################################################



########### IMPLEMENTATION ####################################################
function gsa(f, method::Shapley, input_distribution::SklarDist;batch=false)

    # Extract variables from shapley structure
    n_perms = method.n_perms
    n_boot = method.n_boot
    n_var = method.n_var
    n_outer = method.n_outer
    n_inner = method.n_inner
    dim = method.dim

    # determine if you are running rand_perm or exact_perm version of the algorithm
    if (n_perms==-1)
        estimation_method = "exact";
        perms = collect(permutations(range(1,dim), dim));
        n_perms = length(perms);
    else
        estimation_method = "random";
        perms = [randperm(dim) for i in range(1, n_perms)]
    end

    # Creation of the design matrix
    sample_A = copy(transpose(rand(input_distribution, n_var))); # number of samples x number of features 
    sample_B = zeros((n_perms * (dim - 1) * n_outer * n_inner, dim));

    #---> iterate to create the sample 
    for (i_p, perm) ∈ collect(enumerate(perms))
        idx_perm_sorted = sortperm(perm) # Sort the variable ids

        for j in 1:(dim-1)
            
            # normal set
            idx_plus = perm[1:j];
            # Complementary set
            idx_minus = perm[j+1:end];
            sample_complement = sample_subset(input_distribution, n_outer, idx_minus);

            for  l in range(1,size(sample_complement)[1])
                curr_sample = sample_complement[l, :];

                # Sampling of the set conditionally to the complementary element 
                xj = cond_sampling(input_distribution, n_inner, idx_plus, idx_minus, curr_sample);
                xx = reduce(hcat, (xj, repeat(transpose(curr_sample), n_inner)));
                ind_inner = (i_p - 1) * (dim - 1) * n_outer * n_inner + (j-1) * n_outer * n_inner + (l-1) * n_inner; # subtract 1 from all indices
                ind_inner += 1;
                sample_B[ind_inner:ind_inner + n_inner - 1, :] = @view xx[:, idx_perm_sorted]
            end 
        end
    end

    # define the input sample
    X = cat(sample_A, sample_B, dims=1); 

    if batch
        output_sample = f(X);
    else
        f_non_batch = X -> [f(X[j, :]) for j in axes(X, 1)];
        output_sample = f_non_batch(X);
    end

    output_sample_A = output_sample[1:n_var]
    output_sample_B = permutedims(reshape(output_sample[n_var+1:end], (1,n_inner,n_outer,dim-1,n_perms)), (5,4,3,2,1))
    self_output_sample_B = copy(output_sample_B)
    # <----

    #---> compute indices now 
    shapley_indices = zeros(dim, n_boot, 1);
    ξ = zeros(n_perms, dim, n_boot, 1); # estimation of the cost function (Var[Y] - E[Var[Y|Xj]])

    variance = zeros(n_boot, 1)

    # The first iteration is computed over the all sample.
    idx_for_var = range(1, n_var);
    idx_for_cond_var = range(1, n_outer);
    var_y = var(output_sample_A[idx_for_var]);
    variance[1] = var_y;
    # Conditional variances
    output_sample_B = self_output_sample_B[:, :, idx_for_cond_var, :, :];
    conditional_variance = var(output_sample_B, dims=4);
    conditional_variance = dropdims(conditional_variance; dims=4); # need to squeeze the dimension over which we applied the variance operator. Julia does not do it automatically
    # conditional_variance is the same 
    mean_conditional_variance = mean(conditional_variance, dims=3);
    mean_conditional_variance = dropdims(mean_conditional_variance; dims=3); # need to squeeze the dimension over which we applied the mean operator. Julia does not do it automatically
    # # Cost estimation
    ξ[:, :, 1] .= dropdims(reduce(hcat, (mean_conditional_variance, repeat([var_y], n_perms))), dims=3) ;


    # Allocations for remaining Bootstrap samples 
    idx_for_var = similar(rand(1:n_var, n_var));
    #range_of = 1:n_var;
    idx_for_cond_var = similar(rand(1:n_outer, n_outer));


    for i in range(2, n_boot)
        
        # Bootstrap sample indexes
        rand!(idx_for_var, 1:n_var );
        rand!(idx_for_cond_var, 1:n_outer)

        var_y = var(output_sample_A[idx_for_var]);
        variance[i] = var_y;
        # Conditional variances
        output_sample_B = @view self_output_sample_B[:, :, idx_for_cond_var, :, :];
        conditional_variance = var(output_sample_B, dims=4);
        conditional_variance = dropdims(conditional_variance; dims=4); # need to squeeze the dimension over which we applied the variance operator. Julia does not do it automatically
        mean_conditional_variance = mean(conditional_variance, dims=3);
        mean_conditional_variance = dropdims(mean_conditional_variance; dims=3); # need to squeeze the dimension over which we applied the mean operator. Julia does not do it automatically
        # # Cost estimation
        ξ[:, :, i] .= dropdims(reduce(hcat, (mean_conditional_variance, repeat([var_y], n_perms))), dims=3) ;
        
    end
    

    ξ[:, 2:end, :, :]  .= ξ[:, 2:end, :, :] - ξ[:, 1:end-1, :, :]; 

    # Cost variation
    for i in range(1, n_boot)

        tmp_perms = perms;
        
        # estimate shapley 
        for i_p in range(1, length(tmp_perms))
            perm = perms[i_p];
            @views shapley_indices[perm, i] .+=  ξ[i_p, :, i];
            
        end
    end

    output_variance = reshape(variance, (1, size(variance)[1], size(variance)[2]));
    shapley_indices .= shapley_indices ./ n_perms ./ output_variance;

    return ShapleyResult(shapley_indices, output_variance)

    
end



@doc raw"""

    Shapley(n_perms, n_var, n_outer, n_inner)

- `n_perms`: number of permutations to consider. Defaults to -1, which means all permutations
            are considered hence the exact Shapley effects
            algorithm is used. If `n_perms` is set to a positive integer, then the random version
            of Shapley effects is used.
- `n_var`: size of each bootstrapped sample
- `n_outer`: number of samples to be taken to estimate conditional variance
- `n_inner`: size of each `n_outer` sample taken

## Method Details

Shapely effects is a variance based method to assign attribution
to each feature based on how sentitive the function to the feature.
Shapley effects take into account that features could be dependent, which
is not possible in previous methods like Sobol indices. In our implementation,
we use Copulas.jl to define the joint input distribution as a SklarDist.

## API

    gsa(f, method::Shapley, input_distribution::SklarDist; batch=false)


### Example

```julia
using Copulas, Distributions, GlobalSensitivity

function ishi(X)
    A = 7
    B = 0.1
    sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1])
end

function ishi_batch(X)
    A = 7
    B = 0.1
    @. sin(X[1, :]) + A*sin(X[2, :])^2+ B*X[3, :]^4 *sin(X[1, :])
end

n_perms = -1; # -1 indicates that we want to consider all permutations. One can also use n_perms > 0
n_var = 1000;
n_outer = 100;
n_inner = 3

dim = 3;
margins = (Uniform(-pi, pi), Uniform(-pi, pi), Uniform(-pi, pi));
dependency_matrix = Matrix(I, dim, dim)

C = GaussianCopula(dependency_matrix);
input_distribution = SklarDist(C,margins);

method = Shapley(n_perms=n_perms, n_var = n_var, n_outer = n_outer, n_inner = n_inner);

###### non-batch
result_non_batch = gsa(ishi,method,input_distribution,batch=false)
shapley_effects = result_non_batch.shapley_effects
println(shapley_effects)

###### batch
result_batch = gsa(ishi_batch,method,input_distribution,batch=true)
shapley_effects = result_batch.shapley_effects
println(shapley_effects)

#### Example with correlated inputs
d = 3
mu = zeros(d)
sig = [1, 1, 2]
ro = 0.9
Cormat = [1 0 0; 0 1 ro; 0 ro 1]
Covmat = (sig * transpose(sig)) .* Cormat

margins = [Normal(mu[i], sig[i]) for i in 1:d]
copula = GaussianCopula((sig * transpose(sig)) .* Cormat)
input_distribution = SklarDist(copula, margins)

result = gsa(ishi, method, input_distribution, batch = false)
```
"""
struct Shapley <: GSAMethod
    n_perms::Int
    n_var::Int
    n_outer::Int
    n_inner::Int
end

function Shapley(; n_perms = -1, n_var, n_outer, n_inner = 3)
    Shapley(n_perms, n_var, n_outer, n_inner)
end

mutable struct ShapleyResult{T1, T2}
    shapley_effects::T1
    std_err::T2
    CI_lower::T1
    CI_upper::T1
end

function find_cond_mean_var(cov::Matrix,
        dependent_ind::Vector{Int},
        given_ind::Vector{Int},
        X_given::Vector)
    """
    Find the conditional mean and variance of the given distribution
    """

    mat_b = cov[dependent_ind, dependent_ind]

    mat_c = cov[given_ind, dependent_ind]

    mat_d = cov[given_ind, given_ind]

    mat_cdinv = transpose(mat_c) / mat_d
    conditional_mean = mat_cdinv * X_given
    conditional_var = mat_b - mat_cdinv * mat_c

    return conditional_mean, conditional_var
end

function cond_sampling(distribution::SklarDist{<:IndependentCopula},
        n_sample::Int,
        idx::Vector{Int},
        idx_c::Vector{Int},
        x_cond::AbstractArray)
    # conditional sampling in independent random vector is just subset sampling.
    samples = zeros(eltype(x_cond), length(idx), n_sample)
    rand!(Copulas.subsetdims(distribution, idx), samples)
    return samples
end

function cond_sampling(distribution::SklarDist{<:GaussianCopula},
        n_sample::Int,
        idx::Vector{Int},
        idx_c::Vector{Int},
        x_cond::AbstractArray)

    # select the correct marginal distributions for the two subsets of features
    margins_dependent = [distribution.m[Int(i)] for i in idx]
    margins_conditional = [distribution.m[Int(i)] for i in idx_c]

    # create a conditioned variable that follows a normal distribution
    cond_norm_var = zeros(eltype(x_cond), size(x_cond))
    for (i, marginal) in collect(enumerate(margins_conditional))
        cond_norm_var[i] = quantile.(Normal(zero(eltype(x_cond))), cdf(marginal, x_cond[i]))
    end

    corr_mat = distribution.C.Σ
    cond_mean, cond_var = find_cond_mean_var(corr_mat, idx, idx_c, cond_norm_var)

    n_dep = length(idx)

    # need to sample from univariate normal and multivariate normal using different functions
    samples = zeros(eltype(cond_mean), n_dep, n_sample)
    if n_dep == 1
        dist_cond = Normal(cond_mean[1, 1], sqrt(cond_var[1, 1]))
    else
        dist_cond = MvNormal(cond_mean, Symmetric(cond_var))
    end
    Random.rand!(dist_cond, samples)

    std_norm = Normal(zero(eltype(cond_mean)))
    samples .= quantile.(margins_dependent, cdf.(std_norm, samples))

    return samples
end
##############################################################################

########### IMPLEMENTATION ####################################################
function gsa(f, method::Shapley, input_distribution::SklarDist; batch = false)

    # Extract variables from shapley structure
    n_perms = method.n_perms
    n_var = method.n_var
    n_outer = method.n_outer
    n_inner = method.n_inner
    dim = length(input_distribution)

    # determine if you are running rand_perm or exact_perm version of the algorithm
    if (n_perms == -1)
        @info "Since `n_perms` wasn't set the exact version of Shapley will be used"
        perms = collect(permutations(1:dim, dim))
        n_perms = length(perms)
    else
        @info "Since `n_perms` was set the random version of Shapley will be used"
        perms = [randperm(dim) for i in 1:n_perms]
    end

    # Creation of the design matrix
    sample_A = rand(input_distribution, n_var) # number of samples x number of features
    sample_B = zeros(eltype(sample_A), dim, n_perms * (dim - 1) * n_outer * n_inner)

    #---> iterate to create the sample
    for (i_p, perm) in collect(enumerate(perms))
        idx_perm_sorted = sortperm(perm) # Sort the variable ids

        for j in 1:(dim - 1)

            # normal set
            idx_plus = perm[1:j]
            # Complementary set
            idx_minus = perm[(j + 1):end]
            sample_complement = rand(
                Copulas.subsetdims(input_distribution, idx_minus), n_outer)

            if size(sample_complement, 2) == 1
                sample_complement = reshape(
                    sample_complement, (1, length(sample_complement)))
            end

            for l in 1:n_outer
                curr_sample = @view sample_complement[:, l]
                # Sampling of the set conditionally to the complementary element
                xj = cond_sampling(input_distribution,
                    n_inner,
                    idx_plus,
                    idx_minus,
                    curr_sample)
                xx = [xj; repeat(curr_sample, 1, size(xj, 2))]
                ind_inner = (i_p - 1) * (dim - 1) * n_outer * n_inner +
                            (j - 1) * n_outer * n_inner + (l - 1) * n_inner # subtract 1 from all indices
                ind_inner += 1
                sample_B[:, ind_inner:(ind_inner + n_inner - 1)] = @view xx[
                    idx_perm_sorted,
                    :]
            end
        end
    end

    # define the input sample
    X = [sample_A sample_B]

    if batch
        output_sample = f(X)
        multioutput = output_sample isa AbstractMatrix
        y_size = nothing
    else
        f_non_batch = X -> [f(X[:, j]) for j in axes(X, 2)]
        output_sample = f_non_batch(X)
        multioutput = !(eltype(output_sample) <: Number)
        if eltype(output_sample) <: RecursiveArrayTools.AbstractVectorOfArray
            y_size = size(output_sample[1])
            output_sample = vec.(output_sample)
        else
            y_size = nothing
        end
    end

    if !multioutput
        Sh = zeros(dim)
        Sh2 = zeros(dim)

        m = size(perms, 1)
        Y = output_sample[1:(n_var)]
        y = output_sample[(n_var + 1):end]
        VarY = var(Y)
        for p in 1:m
            perm = perms[p]
            prevC = 0
            for j in 1:dim
                if j == dim
                    Chat = VarY
                    Δ = Chat - prevC
                else
                    cVar = map(1:n_outer) do l
                        Y = @view y[1:(n_inner)]
                        y = @view y[(n_inner + 1):end]
                        var(Y)
                    end
                    Chat = mean(cVar)
                    Δ = Chat - prevC
                    Δ2 = mean((cVar .- prevC) .^ 2) - Δ^2
                    Sh2[perm[j]] += Δ2
                end
                Sh[perm[j]] += Δ
                prevC = Chat
            end
        end
        Sh = Sh / m / VarY
        Sh2 = Sh2 / m / VarY^2
        ShSE = sqrt.(Sh2 / n_outer)
        ShCI = [Sh - 1.96 * ShSE, Sh + 1.96 * ShSE]
    else
        output_dim = length(output_sample[1])
        Sh = zeros(output_dim, dim)
        Sh2 = zeros(output_dim, dim)

        m = size(perms, 1)
        Y = reduce(hcat, output_sample[1:(n_var)])
        y = reduce(hcat, output_sample[(n_var + 1):end])
        VarY = var(Y, dims = 2)

        for p in 1:m
            perm = perms[p]
            prevC = zeros(output_dim)
            for j in 1:dim
                if j == dim
                    Chat = VarY
                    Δ = Chat - prevC
                else
                    cVar = mapreduce(hcat, 1:n_outer) do l
                        Y = @view y[:, 1:(n_inner)]
                        y = @view y[:, (n_inner + 1):end]
                        var(Y, dims = 2)
                    end
                    Chat = mean(cVar, dims = 2)
                    Δ = Chat - prevC
                    Δ2 = mean((cVar .- prevC) .^ 2, dims = 2) - Δ .^ 2
                    Sh2[:, perm[j]] += Δ2
                end
                Sh[:, perm[j]] += Δ
                prevC = Chat
            end
        end
        Sh = Sh ./ m ./ VarY
        Sh2 = Sh2 ./ m ./ VarY .^ 2
        ShSE = sqrt.(Sh2 ./ n_outer)
        ShCI = [Sh - 1.96 .* ShSE, Sh + 1.96 .* ShSE]
        if y_size !== nothing
            f_shape = let y_size = y_size
                x -> [reshape(x[:, i], y_size) for i in 1:size(x, 2)]
            end
            Sh = f_shape(Sh)
            Sh2 = f_shape(Sh2)
            ShSE = f_shape(ShSE)
            ShCI = f_shape.(ShCI)
        end
    end
    return ShapleyResult(Sh, ShSE, ShCI[1], ShCI[2])
end

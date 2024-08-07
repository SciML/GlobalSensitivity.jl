@doc raw"""

MutualInformation(; order = [0, 1], nboot = 1, conf_level = 0.95, n_bin_configurations = 800, n_samples_per_configuration = 100)

- `order`: A vector of integers specifying the order of sensitivity indices to be calculated. Default is `[0, 1]`. Possible values
    are `[0]` for total order indices, `[1]` for first order indices, and `[2]` for second order indices.
- `nboot`: Number of bootstraps to be used for confidence interval estimation. Default is `1`.
- `conf_level`: Confidence level for the confidence interval estimation. Default is `0.95`.
- `n_bin_configurations`: Number of bin configurations to be used for discretization entropy estimation. Default is `800`.
- `n_samples_per_configuration`: Number of samples per bin configuration, used for estimation of discretization entropy. 
   Default is `100`.


## Method Details

The sensitivity analysis based on mutual information is an alternative approach to sensitivity analysis based on information theoretic measures. In this method,
the output uncertainty is quantified by the entropy of the output distribution, instead of taking a variance-based approach. The Shannon entropy of the output is 
given by:

```math
H(Y) = -\sum_y p(y) \log p(y)
```
Where ``p(y)`` is the probability density function of the output ``Y``. By fixing an input ``X_i``, the conditional entropy of the output ``Y`` is given by:

```math
H(Y|X_i) = -\sum_{x} p(x) H(Y|X_i = x)
```

The mutual information between the input ``X_i`` and the output ``Y`` is then given by:

```math
I(X_i;Y) = H(Y) - H(Y|X_i) = H(X) + H(Y) - H(X,Y)
```

Where ``H(X,Y)`` is the joint entropy of the input and output. The mutual information can be used to calculate the sensitivity indices of the input parameters. 

### First Order Sensitivity Indices
The first order sensitivity indices are calculated as the mutual information between the input ``X_i`` and the output ``Y`` divided by the entropy of the output ``Y``:

```math
S_{1,i} = \frac{I(X_i;Y)}{H(Y)}
```

This measure is introduced in Lüdtke et al. (2007)[^1] and also present in Datseris & Parlitz (2022)[^2] in an unnormalized form.

### Second Order Sensitivity Indices
To account for the interaction between input parameters, and their effect on the output, the second order sensitivity indices can be calculated using the 
conditional mutual information between two input parameters ``X_i`` and ``X_j`` given the output ``Y``. They have to be corrected for the correlation between the input parameters. The
second order sensitivity indices according to Lüdtke et al. (2007)[^1] are given by:

```math
S_{2,i} = \frac{I(X_i;X_j|Y) - I(X_i;X_j)}{H(Y)}
```

### Total Order Sensitivity Indices
From Lüdtke et al. (2007)[^1], the total order sensitivity indices can be calculated as:

```math
S_{\text{total},i} = \frac{H(Y) - H(Y|\{X_1,...,X_n\} \\ X_i)}{H(Y) - H_{\Delta}}}
```

Where ``H_{\Delta}`` is the discretization entropy of the output, which quantifies the remaining uncertainty in the output after discretization of the input parameters.
Estimation of the discretization entropy is done by randomly selecting `n_bin_configurations` number of possible combinations of bins in X. Then, for each configuration,
we have a set of bins ``\{B_{1,j}, B_{2,j},...,B_{n,j}\}`` where ``B_{i,j}`` is the bin for the input parameter ``X_i`` for configuration ``j``. Within each configuration,
`n_samples_per_configuration` number of samples are drawn from the input space where ``\{X_1 = x_1 \in B_{1,j}, X_2 = x_2 \in B_{2,j},...,X_n = x_n \in B_{n,j} \}`` and the 
entropy of the output is calculated. The estimated discretization entropy is then:
```math
\hat{H}_{\Delta} = \frac{1}{n_{\text{config}}} \sum_{j=1}^{n_{\text{config}}} H(Y|\{X_1 = x_1 \in B_{1,j}, X_2 = x_2 \in B_{2,j},...,X_n = x_n \in B_{n,j} \})
```

As a result, as ``n_{\text{config}}`` increases towards the number of possible configurations ``N = n_{\text{bins}}^n_{\text{params}}``, the estimated discretization entropy 
converges to the true discretization entropy. In practice, the number of configurations that need to be sampled was found to be much smaller than the total number of possible 
configurations. [^1]

## API

    gsa(f, method::MutualInformation, p_range; samples, batch = false)

Returns a `MutualInformationResult` object containing the resulting sensitivity indices for the parameters and the corresponding confidence intervals.
The `MutualInformationResult` object contains the following fields:
- `S1`: First order sensitivity indices.
- `S1_Conf_Int`: Confidence intervals for the first order sensitivity indices.
- `S2`: Second order sensitivity indices.
- `S2_Conf_Int`: Confidence intervals for the second order sensitivity indices.
- `ST`: Total order sensitivity indices.
- `ST_Conf_Int`: Confidence intervals for the total order sensitivity indices.

For fields that are not calculated, the corresponding field in the result will be an array of zeros.

### Example

```julia
using GlobalSensitivity

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

lb = -ones(4)*π
ub = ones(4)*π

res1 = gsa(ishi,MutualInformation(),[[lb[i],ub[i]] for i in 1:4],samples=15000)
res2 = gsa(ishi_batch,MutualInformation(),[[lb[i],ub[i]] for i in 1:4],samples=15000,batch=true)
```

### References
[^1]: Lüdtke, N., Panzeri, S., Brown, M., Broomhead, D. S., Knowles, J., Montemurro, M. A., & Kell, D. B. (2007). Information-theoretic sensitivity analysis: a general method for credit assignment in complex networks. Journal of The Royal Society Interface, 5(19), 223–235.
[^2]: Datseris, G., & Parlitz, U. (2022). Nonlinear Dynamics, Ch. 7, pg. 105-119.
"""
struct MutualInformation <: GSAMethod
    order::Vector{Int}
    nboot::Int
    conf_level::Real
    n_bin_configurations::Int
    n_samples_per_configuration::Int
end

function MutualInformation(; order = [0, 1], nboot = 1, conf_level = 0.95,
        n_bin_configurations = 800, n_samples_per_configuration = 100)
    MutualInformation(
        order, nboot, conf_level, n_bin_configurations, n_samples_per_configuration)
end

struct MutualInformationResult{T1, T2, T3, T4}
    S1::T1
    S1_Conf_Int::T2
    S2::T3
    S2_Conf_Int::T4
    ST::T1
    ST_Conf_Int::T2
end

function _total_order_ci(
        Xi, Y, entropy_Y, discretization_entropy; n_bootstraps = 100, conf_level = 0.95)
    # perform permutations of Y and calculate total order
    mi_values = zeros(n_bootstraps)
    est = ValueHistogram(Int(round(sqrt(length(Y)))))
    Y_perm = copy(Y)
    entropy_Xi = entropy(est, Xi)
    for i in 1:n_bootstraps
        shuffle!(Y_perm)

        conditional_entropy = entropy(est, StateSpaceSet(Xi, Y_perm)) - entropy_Xi
        mi_values[i] = (entropy_Y - conditional_entropy) /
                       (entropy_Y - discretization_entropy)
    end

    α = 1 - conf_level

    return quantile(mi_values, [α / 2, 1 - α / 2])
end

function _first_order_ci(Xi, Y; n_bootstraps = 100, conf_level = 0.95)

    # perform permutations of Y and calculate mutual information
    mi_values = zeros(n_bootstraps)
    est = ValueHistogram(Int(round(sqrt(length(Y)))))
    Y_perm = copy(Y)
    entropy_Xi = entropy(est, Xi)
    entropy_Y = entropy(est, Y)
    for i in 1:n_bootstraps
        shuffle!(Y_perm)
        mutual_information = entropy_Xi + entropy_Y -
                             entropy(est, StateSpaceSet(Xi, Y_perm))
        mi_values[i] = mutual_information / entropy_Y
    end

    α = 1 - conf_level

    return quantile(mi_values, [α / 2, 1 - α / 2])
end

function _second_order_ci(Xi, Xj, Y; n_bootstraps = 100, conf_level = 0.95)

    # perform permutations of Y and calculate second order mutual information
    mi_values = zeros(n_bootstraps)
    est = ValueHistogram(Int(round(sqrt(length(Y)))))
    Y_perm = copy(Y)
    mutual_information = entropy(est, Xi) + entropy(est, Xj) -
                         entropy(est, StateSpaceSet(Xi, Xj))
    entropy_Y = entropy(est, Y_perm)
    for i in 1:n_bootstraps
        shuffle!(Y_perm)
        conditional_mutual_information = entropy(est, StateSpaceSet(Xi, Y_perm)) +
                                         entropy(est, StateSpaceSet(Xj, Y_perm)) -
                                         entropy(est, StateSpaceSet(Xi, Xj, Y_perm)) -
                                         entropy_Y
        mi_values[i] = (conditional_mutual_information - mutual_information) / entropy_Y
    end

    α = 1 - conf_level

    return quantile(mi_values, [α / 2, 1 - α / 2])
end

function _discretization_entropy(X::AbstractArray, f, batch; n_bin_configurations = 800,
        n_samples_per_configuration = 100)
    n_bins = Int(round(sqrt(size(X, 2))))
    n_dims = size(X, 1)

    span = (maximum(X, dims = 2) .- minimum(X, dims = 2)) ./ n_bins

    entropy_Y = 0.0
    for _ in 1:n_bin_configurations
        config = rand(1:n_bins, n_dims)
        bin_edges = [span .* (config .- 1) span .* config]
        if batch
            samples_Y = f(hcat([rand.(Uniform.(bin_edges[:, 1], bin_edges[:, 2]))
                                for _ in 1:n_samples_per_configuration]...))
        else
            samples_Y = [f(rand.(Uniform.(bin_edges[:, 1], bin_edges[:, 2])))
                         for _ in 1:n_samples_per_configuration]
        end
        entropy_Y += entropy(ValueHistogram(Int(round(sqrt(length(samples_Y))))), samples_Y)
    end

    return entropy_Y / n_bin_configurations
end

function _compute_mi(X::AbstractArray, f, batch::Bool, method::MutualInformation)
    discretization_entropy = _discretization_entropy(
        X, f, batch; n_bin_configurations = method.n_bin_configurations,
        n_samples_per_configuration = method.n_samples_per_configuration)
    if batch
        Y = f(X)
        multioutput = Y isa AbstractMatrix
    else
        Y = [f(X[:, j]) for j in axes(X, 2)]
        multioutput = !(eltype(Y) <: Number)
        if eltype(Y) <: RecursiveArrayTools.AbstractVectorOfArray
            y_size = size(Y[1])
            Y = vec.(Y)
            desol = true
        end
    end

    # K is the number of variables, samples is the number of simulations
    K = size(X, 1)

    if method.nboot > size(X, 2)
        throw(ArgumentError("Number of bootstraps must be less than or equal to the number of samples"))
    end
    est = ValueHistogram(Int(round(sqrt(size(Y, 1)))))
    entropy_Y = entropy(est, Y)

    total_order = zeros(K)
    total_order_ci = zeros(K, 2)

    first_order = zeros(K)
    first_order_ci = zeros(K, 2)

    second_order = zeros(K, K)
    second_order_ci = zeros(K, K, 2)

    # calculate total order
    if 0 in method.order
        @inbounds for i in 1:K
            Xi = @view X[i, :]
            conditional_entropy = entropy(est, StateSpaceSet(Xi, Y)) - entropy(est, Xi)
            total_order[i] = (entropy_Y - conditional_entropy) /
                             (entropy_Y - discretization_entropy)
            total_order_ci[i, :] = _total_order_ci(
                Xi, Y, entropy_Y, discretization_entropy,
                n_bootstraps = method.nboot, conf_level = method.conf_level)
        end
    end

    if 1 in method.order
        # calculate mutual information
        @inbounds for i in 1:K
            Xi = @view X[i, :]
            mutual_information = entropy(est, Xi) + entropy_Y -
                                 entropy(est, StateSpaceSet(Xi, Y))
            first_order[i] = mutual_information / entropy_Y
            first_order_ci[i, :] .= _first_order_ci(
                Xi, Y, n_bootstraps = method.nboot, conf_level = method.conf_level)
        end
    end

    if 2 in method.order
        for (i, j) in combinations(1:K, 2)
            Xi = @view X[i, :]
            Xj = @view X[j, :]
            conditional_mutual_information = entropy(est, StateSpaceSet(Xi, Y)) +
                                             entropy(est, StateSpaceSet(Xj, Y)) -
                                             entropy(est, StateSpaceSet(Xi, Xj, Y)) -
                                             entropy_Y
            mutual_information = entropy(est, Xi) + entropy(est, Xj) -
                                 entropy(est, StateSpaceSet(Xi, Xj))
            second_order[i, j] = second_order[j, i] = (conditional_mutual_information -
                                                       mutual_information) / entropy_Y
            second_order_ci[i, j, :] = second_order_ci[j, i, :] = _second_order_ci(
                Xi, Xj, Y, n_bootstraps = method.nboot, conf_level = method.conf_level)
        end
    end

    total_order, total_order_ci, first_order, first_order_ci, second_order, second_order_ci
end

function gsa(f, method::MutualInformation, p_range; samples, batch = false)
    lb = [i[1] for i in p_range]
    ub = [i[2] for i in p_range]

    X = QuasiMonteCarlo.sample(samples, lb, ub, QuasiMonteCarlo.SobolSample())
    total_order, total_order_ci, first_order, first_order_ci, second_order, second_order_ci = _compute_mi(
        X, f, batch, method)

    return MutualInformationResult(first_order, first_order_ci, second_order,
        second_order_ci, total_order, total_order_ci)
end

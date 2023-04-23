@doc raw"""

    Shapley(; method = "monte_carlo", iters = 1024, M = 2048, v = 1, seed = 999, conf_level = 0.95)

- `method`: the type of shapley method you want to use. Can decide between monte carlo and kernel shap
- `iters`: [monte carlo specific] number of iterations
- `M`: [kernel shap specific] The multiplicity of sample data points per coalition.
- `v`: [kernel shap specific] The number of distinct coalitions to use. It cannot excede `floor(N/2)` where `N` is the number of features.
- `seed`: seed for the random number generator 
- `nboot`: for confidence interval calculation `nboot` should be specified for the number (>0) of bootstrap runs.
- `conf_level`: the confidence level, the default for which is 0.95.

### Example
```julia
using GlobalSensitivity, QuasiMonteCarlo

function test_func_ishi(A:: Number, B:: Number, batch)
    if batch
        f(X::Matrix) = @. sin(X[:, 1]) + A * sin(X[:, 2])^2 + B * X[:, 3]^4 * sin(X[:, 1]);
    else
        f(X::Array) = sin(X[1]) + A * sin(X[2])^2 + B * X[3]^4 * sin(X[1]);
    end

    return f
end

samples = 600000;
p_range = [(-pi, pi), (-pi, pi), (-pi, pi), (-pi, pi)];
f = test_func_ishi(7, 0.3, true);

method = Shapley(method="monte_carlo")
result = gsa(f, method, p_range; samples)

```

"""

struct Shapley <: GSAMethod
    method::String
    iters::Int
    M::Int
    v::Int
    seed::Int
    conf_level::Float
    nboot::Int
end

Shapley(; method = "monte_carlo", iters = 1024, M = 2048, v = 1, seed = 999) = Shapley(method, iters, M, v, seed)

mutable struct ShapleyResult{T1, T2, T3}
    Shapley_indices::T1
    Shapley_indices_ci_lower::T2
    Shapley_indices_ci_upper::T3
end

function gsa(f, method::Shapley, X, res)
    
    #---> define your random number generator 
    rng = MersenneTwister(method.seed);

    #---> initialize your method
    if method == "monte_carlo"
        method = Shapley.MonteCarlo(res, method.iters, rng);
    elseif method == "kernel_shap"
        method =  Shapley.KernelSHAP(res, method.v, method.M, rng);
    end 

    #---> handle batch and non-batch separately 
    if batch
        f_for_shapley = f;
    else
        f_for_shapley = X -> [f(X[j, :]) for j in axes(X, 1)];
    end

    #---> return the shapely values 
    out_as_table = shapley(f1, method, X)
    out_as_matrix = Tables.matrix(out_as_table);

    #---> perform boostraping to find the 95% CI for the median of shapley values for each feature
    stats! = []
    lower_ci! = []
    upper_ci! = []
    n_boot = method.nboot;
    cil = method.conf_level;

    for i in range(1, size(X)[2])

        bs = bootstrap(median, out_as_matrix[:, i], BasicSampling(n_boot));
        bci = confint(bs, BasicConfInt(cil));

        push!(lower_ci!, bci[1][1])
        push!(upper_ci!, bci[1][2])
        push!(stats!, bci[1][3])

    end

    #---> define a ShapleyResult object to return 
    ShapleyResult(stats!, lower_ci!, upper_ci!)

    return out

end

function gsa(f, method::Shapley, p_range::AbstractVector; samples, kwargs...)

    #---> shapley expects the input of f to be a matrix
    f_changed(X::Matrix) = f(Array(X))

    #---> from the p_ranges, define your X 
    lb = [i[1] for i in p_ranges];
    ub = [i[2] for i in p_ranges];
    X = QuasiMonteCarlo.sample(samples, lb, ub, QuasiMonteCarlo.SobolSample());

    #---> massage the X so that it fits the requirements for external package Shapley 
    X = copy(transpose(X));
    X_as_table = Tables.table(X);

    #---> what resource you are using?
    res = CPUThreads();

    gsa(f_changed, method, X_as_table, res; kwargs...)
end
@doc raw"""

    Sobol(; order = [0, 1], nboot = 1, conf_level = 0.95)

- `order`: the order of the indices to calculate. Defaults to [0,1], which means the Total and First order indices. Passing 2 enables calculation of the Second order indices as well.
- `nboot`: for confidence interval calculation `nboot` should be specified for the number (>0) of bootstrap runs.
- `conf_level`: the confidence level, the default for which is 0.95.

## Method Details

Sobol is a variance-based method, and it decomposes the variance of the output of
the model or system into fractions which can be attributed to inputs or sets
of inputs. This helps to get not just the individual parameter's sensitivities,
but also gives a way to quantify the affect and sensitivity from
the interaction between the parameters.

```math
 Y = f_0+ \sum_{i=1}^d f_i(X_i)+ \sum_{i < j}^d f_{ij}(X_i,X_j) ... + f_{1,2...d}(X_1,X_2,..X_d)
```

```math
 Var(Y) = \sum_{i=1}^d V_i + \sum_{i < j}^d V_{ij} + ... + V_{1,2...,d}
```

The Sobol Indices are "order"ed, the first order indices given by ``S_i = \frac{V_i}{Var(Y)}``
the contribution to the output variance of the main effect of `` X_i ``. Therefore, it
measures the effect of varying `` X_i `` alone, but averaged over variations
in other input parameters. It is standardized by the total variance to provide a fractional contribution.
Higher-order interaction indices `` S_{i,j}, S_{i,j,k} `` and so on can be formed
by dividing other terms in the variance decomposition by `` Var(Y) ``.

## API

    gsa(f, method::Sobol, p_range::AbstractVector; samples, kwargs...)
    gsa(f, method::Sobol, A::AbstractMatrix{TA}, B::AbstractMatrix;
             batch = false, Ei_estimator = :Jansen1999,
             distributed::Val{SHARED_ARRAY} = Val(false),
             kwargs...) where {TA, SHARED_ARRAY}

`Ei_estimator` can take `:Homma1996`, `:Sobol2007` and `:Jansen1999` for which
  Monte Carlo estimator is used for the `Ei` term. Defaults to `:Jansen1999`. Details for these can be found in the
  corresponding papers:

- `:Homma1996` - [Homma, T. and Saltelli, A., 1996. Importance measures in global sensitivity analysis of nonlinear models. Reliability Engineering & System Safety, 52(1), pp.1-17.](https://www.sciencedirect.com/science/article/abs/pii/0951832096000026)
- `:Sobol2007` - [I.M. Sobol, S. Tarantola, D. Gatelli, S.S. Kucherenko and W. Mauntz, 2007, Estimating the approx- imation errors when fixing unessential factors in global sensitivity analysis, Reliability Engineering and System Safety, 92, 957–960.](https://www.sciencedirect.com/science/article/abs/pii/S0951832006001499) and [A. Saltelli, P. Annoni, I. Azzini, F. Campolongo, M. Ratto and S. Tarantola, 2010, Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index, Computer Physics Communications 181, 259–270.](https://www.sciencedirect.com/science/article/abs/pii/S0010465509003087)
- `:Jansen1999` - [M.J.W. Jansen, 1999, Analysis of variance designs for model output, Computer Physics Communi- cation, 117, 35–43.](https://www.sciencedirect.com/science/article/abs/pii/S0010465598001544)
- `:Janon2014` - [Janon, A., Klein, T., Lagnoux, A., Nodet, M., & Prieur, C. (2014). Asymptotic normality and efficiency of two Sobol index estimators. ESAIM: Probability and Statistics, 18, 342-364.](https://arxiv.org/abs/1303.6451)

### Example

```julia
using GlobalSensitivity, QuasiMonteCarlo

function ishi(X)
    A = 7
    B = 0.1
    sin(X[1]) + A * sin(X[2])^2 + B * X[3]^4 * sin(X[1])
end

samples = 600000
lb = -ones(4) * π
ub = ones(4) * π
sampler = SobolSample()
A, B = QuasiMonteCarlo.generate_design_matrices(samples, lb, ub, sampler)

res1 = gsa(ishi, Sobol(order = [0, 1, 2]), A, B)

function ishi_batch(X)
    A = 7
    B = 0.1
    @. sin(X[1, :]) + A * sin(X[2, :])^2 + B * X[3, :]^4 * sin(X[1, :])
end

res2 = gsa(ishi_batch, Sobol(), A, B, batch = true)
```
"""
struct Sobol <: GSAMethod
    order::Vector{Int}
    nboot::Int
    conf_level::Float64
end

Sobol(; order = [0, 1], nboot = 1, conf_level = 0.95) = Sobol(order, nboot, conf_level)

mutable struct SobolResult{T1, T2, T3, T4}
    S1::T1
    S1_Conf_Int::T2
    S2::T3
    S2_Conf_Int::T4
    ST::T1
    ST_Conf_Int::T2
end

function fuse_designs(A, B; second_order = false)
    d = size(A, 1)
    Aᵦ = [copy(A) for i in 1:d]
    for i in 1:d
        Aᵦ[i][i, :] = @view(B[i, :])
    end
    if second_order
        Bₐ = [copy(B) for i in 1:d]
        for i in 1:d
            Bₐ[i][i, :] = @view(A[i, :])
        end
        return hcat(A, B, reduce(hcat, Aᵦ), reduce(hcat, Bₐ))
    end
    return hcat(A, B, reduce(hcat, Aᵦ))
end

function gsa(
        f, method::Sobol, A::AbstractMatrix{TA}, B::AbstractMatrix;
        batch = false, Ei_estimator = :Jansen1999,
        distributed::Val{SHARED_ARRAY} = Val(false),
        kwargs...
    ) where {TA, SHARED_ARRAY}
    d, n = size(A)
    nboot = method.nboot # load to help alias analysis
    n = n ÷ nboot
    multioutput = false
    Anb = Vector{Matrix{TA}}(undef, nboot)
    for i in 1:nboot
        Anb[i] = A[:, (n * (i - 1) + 1):(n * (i))]
    end
    Bnb = Vector{Matrix{TA}}(undef, nboot)
    for i in 1:nboot
        Bnb[i] = B[:, (n * (i - 1) + 1):(n * (i))]
    end
    _all_points = mapreduce(
        (args...) -> fuse_designs(
            args...;
            second_order = 2 in method.order
        ),
        hcat, Anb, Bnb
    )
    if SHARED_ARRAY && isbits(TA)
        all_points = SharedMatrix{TA}(size(_all_points))
        all_points .= _all_points
    else
        all_points = _all_points
    end

    return if batch
        all_y = f(all_points)
        multioutput = all_y isa AbstractMatrix
        y_size = nothing
        gsa_sobol_all_y_analysis(
            method, all_y, d, n, Ei_estimator, y_size,
            Val(multioutput)
        )
    else
        _y = [f(all_points[:, i]) for i in 1:size(all_points, 2)]
        multioutput = !(eltype(_y) <: Number)
        if eltype(_y) <: RecursiveArrayTools.AbstractVectorOfArray
            y_size = size(_y[1])
            _y = vec.(_y)
        else
            y_size = nothing
        end
        if multioutput
            gsa_sobol_all_y_analysis(
                method, reduce(hcat, _y), d, n, Ei_estimator, y_size,
                Val(true)
            )
        else
            gsa_sobol_all_y_analysis(method, _y, d, n, Ei_estimator, y_size, Val(false))
        end
    end
end
function gsa_sobol_all_y_analysis(
        method, all_y::AbstractArray{T}, d, n, Ei_estimator,
        y_size, ::Val{multioutput}
    ) where {T, multioutput}
    nboot = method.nboot
    Eys = multioutput ? Matrix{T}[] : T[]
    Varys = multioutput ? Matrix{T}[] : T[]
    Vᵢs = multioutput ? Matrix{T}[] : Vector{T}[]
    Vᵢⱼs = multioutput ? Array{T, 3}[] : Matrix{T}[]
    Eᵢs = multioutput ? Matrix{T}[] : Vector{T}[]
    step = 2 in method.order ? 2 * d + 2 : d + 2
    if !multioutput
        for i in 1:step:(step * nboot)
            push!(Eys, mean(all_y[((i - 1) * n + 1):((i + 1) * n)]))
            push!(Varys, var(all_y[((i - 1) * n + 1):((i + 1) * n)]))

            fA = all_y[((i - 1) * n + 1):(i * n)]
            fB = all_y[(i * n + 1):((i + 1) * n)]
            fAⁱ = [all_y[(j * n + 1):((j + 1) * n)] for j in (i + 1):(i + d)]
            if 2 in method.order
                fBⁱ = [all_y[(j * n + 1):((j + 1) * n)] for j in (i + d + 1):(i + 2 * d)]
            end

            push!(Vᵢs, [sum(fB .* (fAⁱ[k] .- fA)) for k in 1:d] ./ n)
            if 2 in method.order
                M = zeros(T, d, d)
                for k in 1:d
                    for j in (k + 1):d
                        M[k, j] = sum((fBⁱ[k] .* fAⁱ[j]) .- (fA .* fB)) / n
                    end
                end
                push!(Vᵢⱼs, M)
            end
            if Ei_estimator === :Homma1996
                push!(
                    Eᵢs,
                    [
                        sum((fA .- (sum(fA) ./ n)) .^ 2) ./ (n - 1) .-
                            sum(fA .* fAⁱ[k]) ./ (n) + (sum(fA) ./ n) .^ 2 for k in 1:d
                    ]
                )
            elseif Ei_estimator === :Sobol2007
                push!(Eᵢs, [sum(fA .* (fA .- fAⁱ[k])) for k in 1:d] ./ (n))
            elseif Ei_estimator === :Jansen1999
                push!(Eᵢs, [sum(abs2, fA - fAⁱ[k]) for k in 1:d] ./ (2n))
            elseif Ei_estimator === :Janon2014
                push!(
                    Eᵢs,
                    [
                        (
                                sum(fA .^ 2 + fAⁱ[k] .^ 2) ./ (2n) .-
                                (sum(fA + fAⁱ[k]) ./ (2n)) .^ 2
                            ) * (
                                1.0 .-
                                (
                                    1 / n .* sum(fA .* fAⁱ[k])
                                    .-
                                    (1 / n .* sum((fA .+ fAⁱ[k]) ./ 2)) .^ 2
                                ) ./
                                (
                                    1 / n .* sum((fA .^ 2 .+ fAⁱ[k] .^ 2) ./ 2) -
                                    (1 / n .* sum((fA .+ fAⁱ[k]) ./ 2)) .^ 2
                                )
                            )
                            for k in 1:d
                    ]
                )
            end
        end
    else
        for i in 1:step:(step * nboot)
            push!(Eys, mean(all_y[:, ((i - 1) * n + 1):((i + 1) * n)], dims = 2))
            push!(Varys, var(all_y[:, ((i - 1) * n + 1):((i + 1) * n)], dims = 2))

            fA = all_y[:, ((i - 1) * n + 1):(i * n)]
            fB = all_y[:, (i * n + 1):((i + 1) * n)]
            fAⁱ = [all_y[:, (j * n + 1):((j + 1) * n)] for j in (i + 1):(i + d)]
            if 2 in method.order
                fBⁱ = [all_y[:, (j * n + 1):((j + 1) * n)] for j in (i + d + 1):(i + 2 * d)]
            end

            push!(
                Vᵢs,
                reduce(hcat, [sum(fB .* (fAⁱ[k] .- fA), dims = 2) for k in 1:d] ./ n)
            )

            if 2 in method.order
                M = zeros(T, d, d, length(Eys[1]))
                for k in 1:d
                    for j in (k + 1):d
                        Vₖⱼ = sum((fBⁱ[k] .* fAⁱ[j]) .- (fA .* fB), dims = 2) / n
                        for l in 1:length(Eys[1])
                            M[k, j, l] = Vₖⱼ[l]
                        end
                    end
                end
                push!(Vᵢⱼs, M)
            end
            if Ei_estimator === :Homma1996
                push!(
                    Eᵢs,
                    reduce(
                        hcat,
                        [
                            sum((fA .- (sum(fA, dims = 2) ./ n)) .^ 2, dims = 2) ./ (n - 1) .-
                                sum(fA .* fAⁱ[k], dims = 2) ./ (n) + (sum(fA, dims = 2) ./ n) .^ 2
                                for k in 1:d
                        ]
                    )
                )
            elseif Ei_estimator === :Sobol2007
                push!(
                    Eᵢs,
                    reduce(
                        hcat,
                        [sum(fA .* (fA .- fAⁱ[k]), dims = 2) for k in 1:d] ./ (n)
                    )
                )
            elseif Ei_estimator === :Jansen1999
                push!(
                    Eᵢs,
                    reduce(hcat, [sum(abs2, fA - fAⁱ[k], dims = 2) for k in 1:d] ./ (2n))
                )
            elseif Ei_estimator === :Janon2014
                push!(
                    Eᵢs,
                    reduce(
                        hcat,
                        [
                            (
                                    sum(fA .^ 2 + fAⁱ[k] .^ 2, dims = 2) ./ (2n) .-
                                    (sum(fA + fAⁱ[k], dims = 2) ./ (2n)) .^ 2
                                ) .* (
                                    1.0 .-
                                    (
                                        1 / n .* sum(fA .* fAⁱ[k], dims = 2) .-
                                        (1 / n * sum((fA .+ fAⁱ[k]) ./ 2, dims = 2)) .^ 2
                                    ) ./
                                    (
                                        1 / n .* sum((fA .^ 2 .+ fAⁱ[k] .^ 2) ./ 2, dims = 2) .-
                                        (1 / n * sum((fA .+ fAⁱ[k]) ./ 2, dims = 2)) .^ 2
                                    )
                                ) for k in 1:d
                        ]
                    )
                )
            end
        end
    end
    if 2 in method.order
        Sᵢⱼs = similar(Vᵢⱼs)
        for i in 1:nboot
            if !multioutput
                M = zeros(T, d, d)
                for k in 1:d
                    for j in (k + 1):d
                        M[k, j] = Vᵢs[i][k] + Vᵢs[i][j]
                    end
                end
                Sᵢⱼs[i] = (Vᵢⱼs[i] - M) ./ Varys[i]
            else
                M = zeros(T, d, d, length(Eys[1]))
                for l in 1:length(Eys[1])
                    for k in 1:d
                        for j in (k + 1):d
                            M[k, j, l] = Vᵢs[i][l, k] + Vᵢs[i][l, j]
                        end
                    end
                end
                Sᵢⱼs[i] = cat(
                    [
                        (Vᵢⱼs[i][:, :, l] - M[:, :, l]) ./ Varys[i][l]
                            for l in 1:length(Eys[1])
                    ]...;
                    dims = 3
                )
            end
        end
    end

    Sᵢs = [Vᵢs[i] ./ Varys[i] for i in 1:nboot]
    Tᵢs = [Eᵢs[i] ./ Varys[i] for i in 1:nboot]
    if nboot > 1
        size_ = size(Sᵢs[1])
        S1 = [[Sᵢ[i] for Sᵢ in Sᵢs] for i in 1:length(Sᵢs[1])]
        ST = [[Tᵢ[i] for Tᵢ in Tᵢs] for i in 1:length(Tᵢs[1])]

        function calc_ci(x, mean = nothing)
            alpha = (1 - method.conf_level)
            return std(x, mean = mean) / sqrt(length(x))
        end
        S1_CI = map(calc_ci, S1)
        ST_CI = map(calc_ci, ST)

        if 2 in method.order
            size__ = size(Sᵢⱼs[1])
            S2_CI = Array{T}(undef, size__)
            Sᵢⱼ = Array{T}(undef, size__)
            b = getindex.(Sᵢⱼs, 1)
            Sᵢⱼ[1] = b̄ = mean(b)
            S2_CI[1] = calc_ci(b, b̄)
            for i in 2:length(Sᵢⱼs[1])
                b .= getindex.(Sᵢⱼs, i)
                Sᵢⱼ[i] = b̄ = mean(b)
                S2_CI[i] = calc_ci(b, b̄)
            end
        end
        Sᵢ = reshape(mean.(S1), size_...)
        Tᵢ = reshape(mean.(ST), size_...)
    else
        Sᵢ = Sᵢs[1]
        Tᵢ = Tᵢs[1]
        if 2 in method.order
            Sᵢⱼ = Sᵢⱼs[1]
        end
    end
    if isnothing(y_size)
        _Sᵢ = Sᵢ
        _Tᵢ = Tᵢ
    else
        f_shape = let y_size = y_size
            x -> [reshape(x[:, i], y_size) for i in 1:size(x, 2)]
        end
        _Sᵢ = f_shape(Sᵢ)
        _Tᵢ = f_shape(Tᵢ)
    end
    return SobolResult(
        _Sᵢ,
        nboot > 1 ? reshape(S1_CI, size_...) : nothing,
        2 in method.order ? Sᵢⱼ : nothing,
        nboot > 1 && 2 in method.order ? S2_CI : nothing,
        _Tᵢ,
        nboot > 1 ? reshape(ST_CI, size_...) : nothing
    )
end

function gsa(f, method::Sobol, p_range::AbstractVector; samples, kwargs...)
    AB = QuasiMonteCarlo.generate_design_matrices(
        samples, [i[1] for i in p_range],
        [i[2] for i in p_range],
        QuasiMonteCarlo.SobolSample(),
        2 * method.nboot
    )
    A = reduce(hcat, @view(AB[1:(method.nboot)]))
    B = reduce(hcat, @view(AB[(method.nboot + 1):end]))
    return gsa(f, method, A, B; kwargs...)
end

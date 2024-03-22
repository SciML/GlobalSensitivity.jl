"""

    eFAST(; num_harmonics::Int = 4)

- `num_harmonics`: the number of harmonics to sum in the Fourier series decomposition, this defaults to 4.

## Method Details

eFAST offers a robust, especially at low sample size, and computationally efficient procedure to
get the first and total order indices as discussed in Sobol. It utilizes monodimensional Fourier decomposition
along a curve, exploring the parameter space. The curve is defined by a set of parametric equations,
```math
x_{i}(s) = G_{i}(sin ω_{i}s), ∀ i=1,2 ,..., N
```
where s is a scalar variable varying over the range ``-∞ < s < +∞``, ``G_{i}`` are transformation functions
and ``{ω_{i}}, ∀ i=1,2,...,N`` is a set of different (angular) frequencies, to be properly selected, associated with each factor for all ``N`` (`samples`) number of parameter sets.
For more details, on the transformation used and other implementation details you can go through [ A. Saltelli et al.](https://dx.doi.org/10.1080/00401706.1999.10485594).

## API

    gsa(f, method::eFAST, p_range::AbstractVector; samples::Int, batch = false,
             distributed::Val{SHARED_ARRAY} = Val(false),
             rng::AbstractRNG = Random.default_rng(), kwargs...) where {SHARED_ARRAY}


### Example

Below we show use of `eFAST` on the Ishigami function.

```julia
using GlobalSensitivity, QuasiMonteCarlo

function ishi(X)
    A= 7
    B= 0.1
    sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1])
end

lb = -ones(4)*π
ub = ones(4)*π

res1 = gsa(ishi,eFAST(),[[lb[i],ub[i]] for i in 1:4],samples=15000)

##with batching
function ishi_batch(X)
    A= 7
    B= 0.1
    @. sin(X[1,:]) + A*sin(X[2,:])^2+ B*X[3,:]^4 *sin(X[1,:])
end

res2 = gsa(ishi_batch,eFAST(),[[lb[i],ub[i]] for i in 1:4],samples=15000,batch=true)

```
"""
struct eFAST <: GSAMethod
    num_harmonics::Int
end

eFAST(; num_harmonics::Int = 4) = eFAST(num_harmonics)

struct eFASTResult{T1}
    S1::T1
    ST::T1
end

function gsa(f, method::eFAST, p_range::AbstractVector; samples::Int, batch = false,
        distributed::Val{SHARED_ARRAY} = Val(false),
        rng::AbstractRNG = Random.default_rng(), kwargs...) where {SHARED_ARRAY}
    @unpack num_harmonics = method
    num_params = length(p_range)
    omega = [(samples - 1) ÷ (2 * num_harmonics)]
    m = omega[1] ÷ (2 * num_harmonics)

    if !(eltype(p_range) <: Distribution)
        dists = [Uniform(p_range[j][1], p_range[j][2]) for j in eachindex(p_range)]
    else
        dists = p_range
    end

    if m >= num_params - 1
        append!(omega, floor.(Int, collect(range(1, stop = m, length = num_params - 1))))
    else
        append!(omega, collect(range(0, stop = num_params - 2)) .% m .+ 1)
    end

    omega_temp = similar(omega)
    s = (2 / samples) * (0:(samples - 1))
    if SHARED_ARRAY
        ps = SharedMatrix{Float64}((num_params, samples * num_params))
    else
        ps = Matrix{Float64}(undef, num_params, samples * num_params)
    end
    @inbounds for i in 1:num_params
        omega_temp[i] = omega[1]
        for k in 1:(i - 1)
            omega_temp[k] = omega[k + 1]
        end
        for k in (i + 1):num_params
            omega_temp[k] = omega[k]
        end
        l = ((i - 1) * samples + 1):(i * samples)
        phi = 2rand(rng)
        for j in 1:num_params
            if !(eltype(p_range) <: Distribution) && p_range[j][1] == p_range[j][2]
                ps[j, l] .= p_range[j][1]
            else
                if eltype(dists) <: UnivariateDistribution
                    ps[j, l] .= quantile.(dists[j],
                        0.5 .+
                        (1 / pi) .*
                        (asin.(sinpi.(omega_temp[j] .* s .+ phi))))
                else
                    ps[j, l] .= quantile(dists[j],
                        0.5 .+
                        (1 / pi) .*
                        (asin.(sinpi.(omega_temp[j] .* s .+ phi))))
                end
            end
        end
    end

    if batch
        all_y = f(ps)
        multioutput = all_y isa AbstractMatrix
        y_size = nothing
        gsa_efast_all_y_analysis(method, all_y, num_params, y_size, samples, omega,
            Val(multioutput))
    else
        _y = [f(ps[:, j]) for j in 1:size(ps, 2)]
        multioutput = !(eltype(_y) <: Number)
        if eltype(_y) <: RecursiveArrayTools.AbstractVectorOfArray
            y_size = size(_y[1])
            __y = vec.(_y)
        else
            y_size = nothing
            __y = _y
        end
        if multioutput
            gsa_efast_all_y_analysis(
                method, reduce(hcat, __y), num_params, y_size, samples,
                omega, Val(true))
        else
            gsa_efast_all_y_analysis(method, __y, num_params, y_size, samples, omega,
                Val(false))
        end
    end
end
function gsa_efast_all_y_analysis(method, all_y, num_params, y_size, samples, omega,
        ::Val{multioutput}) where {multioutput}
    @unpack num_harmonics = method
    if multioutput
        size_ = size(all_y)
        first_order = Vector{Vector{eltype(all_y)}}(undef, num_params)
        total_order = Vector{Vector{eltype(all_y)}}(undef, num_params)
    else
        first_order = Vector{eltype(all_y)}(undef, num_params)
        total_order = Vector{eltype(all_y)}(undef, num_params)
    end
    for i in 1:num_params
        if !multioutput
            ft = (fft(all_y[((i - 1) * samples + 1):(i * samples)]))[2:(samples ÷ 2)]
            ys = abs2.(ft .* inv(samples))
            varnce = 2 * sum(ys)
            first_order[i] = 2 * sum(ys[(1:num_harmonics) * Int(omega[1])]) / varnce
            total_order[i] = 1 .- 2 * sum(ys[1:(omega[1] ÷ 2)]) / varnce
        else
            ys = Vector{Vector{eltype(all_y)}}(undef, size(all_y, 1))
            varnce = Vector{eltype(all_y)}(undef, size(all_y, 1))
            for j in eachindex(ys)
                ff = fft(all_y[j, ((i - 1) * samples + 1):(i * samples)])[2:(samples ÷ 2)]
                ys[j] = ysⱼ = abs2.(ff .* inv(samples))
                varnce[j] = 2 * sum(ysⱼ)
            end
            first_order[i] = map(
                (y, var) -> 2 * sum(y[(1:num_harmonics) * (omega[1])]) ./
                            var, ys, varnce)
            total_order[i] = map((y, var) -> 1 .- 2 * sum(y[1:(omega[1] ÷ 2)]) ./ var, ys,
                varnce)
        end
    end
    if isnothing(y_size)
        _first_order = reduce(hcat, first_order)
        _total_order = reduce(hcat, total_order)
    else
        f_shape = let y_size = y_size
            x -> [reshape(x[:, i], y_size) for i in 1:size(x, 2)]
        end
        _first_order = reduce(hcat, map(f_shape, first_order))
        _total_order = reduce(hcat, map(f_shape, total_order))
    end
    return eFASTResult(_first_order, _total_order)
end

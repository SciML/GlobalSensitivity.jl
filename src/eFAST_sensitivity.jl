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

Note, `p_range` is either a vector of tuples for the upper and lower bound or a vector of `Distribution`s. 

### Example

Below we show use of `eFAST` on the Ishigami function.

```julia
using GlobalSensitivity, QuasiMonteCarlo, Distributions

function ishi(X)
    A= 7
    B= 0.1
    sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1])
end

## define upper and lower limits, a.k.a uniform distributions
lb = -ones(4)*π
ub = ones(4)*π

res1 = gsa(ishi, eFAST(), [[lb[i],ub[i]] for i in 1:4], samples=15000)

# define distributions for the inputs
input_ranges = [Normal(0, 1),
                Uniform(-π, π),
                Uniform(-π, π),
                Uniform(-π, π)]

res2 = gsa(ishi, eFAST(), input_ranges, samples=15000)

## with batching
function ishi_batch(X)
    A= 7
    B= 0.1
    @. sin(X[1,:]) + A*sin(X[2,:])^2+ B*X[3,:]^4 *sin(X[1,:])
end

res3 = gsa(ishi_batch, eFAST(), [[lb[i],ub[i]] for i in 1:4], samples=15000, batch=true)
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

function gsa(
        f, method::eFAST, p_range::AbstractVector; samples::Int, batch = false,
        distributed::Val{SHARED_ARRAY} = Val(false),
        rng::AbstractRNG = Random.default_rng(), kwargs...
    ) where {SHARED_ARRAY}
    (; num_harmonics) = method
    num_params = length(p_range)
    omega = [(samples - 1) ÷ (2 * num_harmonics)]
    m = omega[1] ÷ (2 * num_harmonics)

    if !(eltype(p_range) <: Distribution)
        dists = [Uniform(p_range[j][1], p_range[j][2]) for j in eachindex(p_range)]
    else
        dists = p_range
    end

    n_comp = num_params - 1
    if n_comp == 1
        append!(omega, [1])          # single complementary param → assign frequency 1
    elseif m >= n_comp
        append!(omega, floor.(Int, collect(range(1, stop = m, length = n_comp))))
    else
        append!(omega, collect(range(0, stop = n_comp - 1)) .% m .+ 1)
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
                    ps[j, l] .= quantile.(
                        dists[j],
                        0.5 .+
                            (1 / pi) .*
                            (asin.(sinpi.(omega_temp[j] .* s .+ phi)))
                    )
                else
                    ps[j, l] .= quantile(
                        dists[j],
                        0.5 .+
                            (1 / pi) .*
                            (asin.(sinpi.(omega_temp[j] .* s .+ phi)))
                    )
                end
            end
        end
    end

    return if batch
        all_y = f(ps)
        multioutput = all_y isa AbstractMatrix
        y_size = nothing
        gsa_efast_all_y_analysis(
            method, all_y, num_params, y_size, samples, omega,
            Val(multioutput)
        )
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
                omega, Val(true)
            )
        else
            gsa_efast_all_y_analysis(
                method, __y, num_params, y_size, samples, omega,
                Val(false)
            )
        end
    end
end
function gsa_efast_all_y_analysis(
        method, all_y, num_params, y_size, samples, omega,
        ::Val{multioutput}
    ) where {multioutput}
    (; num_harmonics) = method
    FT = float(eltype(all_y))
    if multioutput
        first_order = Vector{Vector{FT}}(undef, num_params)
        total_order = Vector{Vector{FT}}(undef, num_params)
        nout = size(all_y, 1)
        ft = rfft(reshape(all_y, nout, samples, num_params), 2)
        buf = Vector{FT}(undef, (samples ÷ 2) - 1)
        for (i, fti) in enumerate(eachslice(ft; dims = 3))
            first_order_i = first_order[i] = Vector{FT}(undef, nout)
            total_order_i = total_order[i] = Vector{FT}(undef, nout)
            for (j, row) in enumerate(eachrow(fti))
                map!(abs2, buf, @view(row[2:(samples ÷ 2)]))
                z = sum(buf)
                first_order_i[j] = sum(@view(buf[(1:num_harmonics) * Int(omega[1])])) / z
                total_order_i[j] = 1 - sum(@view(buf[1:(omega[1] ÷ 2)])) / z
            end
        end
    else
        first_order = Vector{FT}(undef, num_params)
        total_order = Vector{FT}(undef, num_params)
        ft = rfft(reshape(all_y, samples, num_params), 1)
        buf = Vector{FT}(undef, (samples ÷ 2) - 1)
        for (i, fti) in enumerate(eachcol(ft))
            map!(abs2, buf, @view(fti[2:(samples ÷ 2)]))
            z = sum(buf)
            first_order[i] = sum(@view(buf[(1:num_harmonics) * Int(omega[1])])) / z
            total_order[i] = 1 - sum(@view(buf[1:(omega[1] ÷ 2)])) / z
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

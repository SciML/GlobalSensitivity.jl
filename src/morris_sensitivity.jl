@doc raw"""

    Morris(; p_steps::Array{Int, 1} = Int[], relative_scale::Bool = false,
                num_trajectory::Int = 10,
                total_num_trajectory::Int = 5 * num_trajectory, len_design_mat::Int = 10)

- `p_steps`: Vector of ``\Delta`` for the step sizes in each direction. Required.
- `relative_scale`: The elementary effects are calculated with the assumption that the parameters lie in the range [0,1] but as this is not always the case scaling is used to get more informative, scaled effects. Defaults to false.
- `total_num_trajectory`, `num_trajectory`: The total number of design matrices that are generated, out of which num_trajectory matrices with the highest spread are used in calculation.
- `len_design_mat`: The size of a design matrix.

## Method Details

The Morris method also known as Morris’s OAT method where OAT stands for
One At a Time can be described in the following steps:

We calculate local sensitivity measures known as “elementary effects”,
which are calculated by measuring the perturbation in the output of the
model on changing one parameter.

```math
EE_i = \frac{f(x_1,x_2,..x_i+ \Delta,..x_k) - y}{\Delta}
```

These are evaluated at various points in the input chosen such that a wide
“spread” of the parameter space is explored and considered in the analysis,
to provide an approximate global importance measure. The mean and variance of
these elementary effects is computed. A high value of the mean implies that
a parameter is important, a high variance implies that its effects are
non-linear or the result of interactions with other inputs. This method
does not evaluate separately the contribution from the
interaction and the contribution of the parameters individually and gives the
effects for each parameter which takes into consideration all the interactions and its
individual contribution.

## API

    gsa(f, method::Morris, p_range::AbstractVector; batch = false,
             rng::AbstractRNG = Random.default_rng(), kwargs...)

### Example

Morris method on Ishigami function

```julia
using GlobalSensitivity

function ishi(X)
    A= 7
    B= 0.1
    sin(X[1]) + A*sin(X[2])^2+ B*X[3]^4 *sin(X[1])
end

lb = -ones(4)*π
ub = ones(4)*π

m = gsa(ishi, Morris(num_trajectory=500000), [[lb[i],ub[i]] for i in 1:4])
```
"""
struct Morris <: GSAMethod
    p_steps::Array{Int, 1}
    relative_scale::Bool
    num_trajectory::Int
    total_num_trajectory::Int
    len_design_mat::Int
end

function Morris(; p_steps::Array{Int, 1} = Int[], relative_scale::Bool = false,
        num_trajectory::Int = 10,
        total_num_trajectory::Int = 5 * num_trajectory, len_design_mat::Int = 10)
    Morris(p_steps, relative_scale, num_trajectory, total_num_trajectory, len_design_mat)
end

struct MatSpread{T1, T2}
    mat::T1
    spread::T2
end

struct MorrisResult{T1, T2}
    means::T1
    means_star::T1
    variances::T1
    elementary_effects::T2
end

function generate_design_matrix(p_range, p_steps, rng; len_design_mat = 10)
    ps = [range(p_range[i][1], stop = p_range[i][2], length = p_steps[i])
          for i in 1:length(p_range)]
    indices = [rand(rng, 1:i) for i in p_steps]
    all_idxs = Vector{typeof(indices)}(undef, len_design_mat)

    for i in 1:len_design_mat
        j = rand(rng, 1:length(p_range))
        indices[j] += (rand(rng) < 0.5 ? -1 : 1)
        if indices[j] > p_steps[j]
            indices[j] -= 2
        elseif indices[j] < 1.0
            indices[j] += 2
        end
        all_idxs[i] = copy(indices)
    end

    B = Array{Array{Float64}}(undef, len_design_mat)
    for j in 1:len_design_mat
        cur_p = [ps[u][(all_idxs[j][u])] for u in 1:length(p_range)]
        B[j] = cur_p
    end
    reduce(hcat, B)
end

function calculate_spread(matrix)
    spread = 0.0
    for i in 2:size(matrix, 2)
        spread += sqrt(sum(abs2.(matrix[:, i] - matrix[:, i - 1])))
    end
    spread
end

function sample_matrices(p_range, p_steps, rng; num_trajectory = 10,
        total_num_trajectory = 5 * num_trajectory, len_design_mat = 10)
    matrix_array = []
    if total_num_trajectory < num_trajectory
        error("total_num_trajectory should be greater than num_trajectory preferably atleast 3-4 times higher")
    end
    for i in 1:total_num_trajectory
        mat = generate_design_matrix(p_range, p_steps, rng; len_design_mat = len_design_mat)
        spread = calculate_spread(mat)
        push!(matrix_array, MatSpread(mat, spread))
    end
    sort!(matrix_array, by = x -> x.spread, rev = true)
    matrices = [i.mat for i in matrix_array[1:num_trajectory]]
    reduce(hcat, matrices)
end

function gsa(f, method::Morris, p_range::AbstractVector; batch = false,
        rng::AbstractRNG = Random.default_rng(), kwargs...)
    (; p_steps, relative_scale, num_trajectory, total_num_trajectory,
        len_design_mat) = method
    if !(length(p_steps) == length(p_range))
        for i in 1:(length(p_range) - length(p_steps))
            push!(p_steps, 100)
        end
    end

    design_matrices = sample_matrices(p_range, p_steps, rng;
        num_trajectory = num_trajectory,
        total_num_trajectory = total_num_trajectory,
        len_design_mat = len_design_mat)

    multioutput = false
    desol = false
    local y_size
    if batch
        all_y = f(design_matrices)
        multioutput = all_y isa AbstractMatrix
    else
        _y = [f(design_matrices[:, i]) for i in 1:size(design_matrices, 2)]
        multioutput = !(eltype(_y) <: Number)
        if eltype(_y) <: RecursiveArrayTools.AbstractVectorOfArray
            y_size = size(_y[1])
            _y = vec.(_y)
            desol = true
        end
        all_y = multioutput ? reduce(hcat, _y) : _y
    end

    effects = []
    for i in 1:num_trajectory
        y1 = multioutput ? all_y[:, (i - 1) * len_design_mat + 1] :
             all_y[(i - 1) * len_design_mat + 1]
        for j in ((i - 1) * len_design_mat + 1):((i * len_design_mat) - 1)
            y2 = y1
            del = design_matrices[:, j + 1] - design_matrices[:, j]
            change_index = 0
            for k in 1:length(del)
                if abs(del[k]) > 0
                    change_index = k
                    break
                end
            end
            del = sum(del)
            y1 = multioutput ? all_y[:, j + 1] : all_y[j + 1]
            if relative_scale == false
                effect = @. (y1 - y2) / (del)
                elem_effect = y1 isa Number ? effect : mean(effect, dims = 2)
            else
                if del > 0
                    effect = @. (y1 - y2) / (y2 * del)
                    elem_effect = y1 isa Number ? effect : mean(effect, dims = 2)
                else
                    effect = @. (y1 - y2) / (y1 * del)
                    elem_effect = y1 isa Number ? effect : mean(effect, dims = 2)
                end
            end
            if length(effects) >= change_index && change_index > 0
                push!(effects[change_index], elem_effect)
            elseif change_index > 0
                while (length(effects) < change_index - 1)
                    push!(effects, typeof(elem_effect)[])
                end
                push!(effects, [elem_effect])
            end
        end
    end
    means = eltype(effects[1])[]
    means_star = eltype(effects[1])[]
    variances = eltype(effects[1])[]
    for k in effects
        if !isempty(k)
            push!(means, mean(k))
            push!(means_star, mean(x -> abs.(x), k))
            push!(variances, var(k))
        else
            push!(means, zero(effects[1][1]))
            push!(means_star, zero(effects[1][1]))
            push!(variances, zero(effects[1][1]))
        end
    end
    if desol
        f_shape = x -> [reshape(x[:, i], y_size) for i in 1:size(x, 2)]
        means = map(f_shape, means)
        means_star = map(f_shape, means_star)
        variances = map(f_shape, variances)
    end
    MorrisResult(reduce(hcat, means), reduce(hcat, means_star), reduce(hcat, variances),
        effects)
end

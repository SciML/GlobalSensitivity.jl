"""
    FractionalFactorial()

`FractionalFactorial` does not have any keyword arguments.

## Method Details

Fractional Factorial method creates a design matrix by utilizing
Hadamard Matrix and uses it to run simulations of the input model.
The main effects are then evaluated by dot product between the contrast
for the parameter and the vector of simulation results. The
corresponding main effects and variance, i.e. square of the main effects,
are returned as results for Fractional Factorial method.

## API

    gsa(f, method::FractionalFactorial; num_params, p_range = nothing, kwargs...)


### Example

```julia
using GlobalSensitivity, Test

f = X -> X[1] + 2 * X[2] + 3 * X[3] + 4 * X[7] * X[12]
res1 = gsa(f,FractionalFactorial(),num_params = 12,samples=10)
```
"""
struct FractionalFactorial <: GSAMethod end

"""
Code based on the theory presetned in:
Saltelli, A. (2008). Global sensitivity analysis: The primer. Chichester: Wiley, pp. 71-76.
"""

using LinearAlgebra

function _recursive_hadamard(k::Integer)
    """
    Generate a hadamard matrix via recursion.
    """
    # base case
    if k == 2
        return [1 1; 1 -1]
    else
        h = _recursive_hadamard(k ÷ 2)
    end

    return hcat(vcat(h, h), vcat(h, -h))
end

function _expanding_window_hadamard(k::Integer)
    """
    Generate hadamard matrix of size k using expanding window approach.
    """
    @assert ispow2(k)

    # intialize
    h = ones(Int64, k, k)
    h[2, 2] = -1

    let
        bot_row = 2
        right_col = 2
        while bot_row < k
            @inbounds cur = h[1:bot_row, 1:right_col]

            new_bot = (bot_row + 1):(bot_row * 2)
            new_right = (right_col + 1):(right_col * 2)
            # update right
            @inbounds h[1:bot_row, new_right] = cur

            # update below
            @inbounds h[new_bot, 1:right_col] = cur

            # update diagonal
            @inbounds h[new_bot, new_right] = -cur

            # update window for each to 'copy' from.
            bot_row *= 2
            right_col *= 2
        end
    end
    return h
end

function _generate_hadamard(k::Integer)
    @assert ispow2(k)
    return _expanding_window_hadamard(k)
end

function generate_ff_design_matrix(num_parameters::Integer)
    """
    param: num_parameters
         The number of parameters to be sampled for.

    return:
         A 2-level fractional factorial design matrix of -1s and 1s with a design of resolution IV.

         The number of rows is 2*number of parameters and the number of columns is the number of parameters.

         See equation (2.31) in Saltelli, A. (2008) for details.
    """

    # If k is not a power of 2, get the next larger number that is
    # so the hadamard matrices can be computed.
    k = Integer(round(2^ceil(log2(num_parameters))))

    s = _generate_hadamard(k)
    design_matrix = vcat(s, -s) # design of resolution IV.

    return design_matrix
end

function generate_ff_sample_matrix(design_matrix::Array{Int64, 2}, levels_list = nothing)
    """

    Convert the desgin matrix to a matrix whose rows can
    be used as inputs to a model by replacing low and high values for each parameter
    with the the low and high end of each parameters range, as specified in levels_list.

    param: design_matrix

          The matrix outputted by 'ff_design' i.e a 2-level fractional
          factorial design matrix of -1s and 1s with a design of resolution IV.
          The number of rows is 2*number of parameters and the
          number of columns is the number of parameters.

    param: levels_list (optional)
        A list of length equal to the number of parameters, where the elements of
        the list are 2-tuples.

        The ith tuple corresponds to the the low-value and the high-value of the ith parameter.
        i.e. levels_list[2][1] corresponds to the low-value of the 2nd parameter,
        and levels_list[2][2] corresponds to the high-value of the 2nd parameter.

        If no list is provided, the low value for each parameter
        is assumed to be 0 and the high value is assumed to be 1.

    returns:
        A matrix with the same dimensions as the design matrix, but with the '-1's replaced with
        the low value specified for each parameter in level_list, and similar the '1's are replaced
        with the high values.

        Each row corresponds to a single input vector to be used as input to a model/function.
    """
    sample_matrix = copy(design_matrix)
    if !isnothing(levels_list)
        for (col_index, (low_value, high_value)) in enumerate(levels_list)
            design_col = @view design_matrix[:, col_index]
            sample_col = @view sample_matrix[:, col_index]

            sample_col[design_col .== -1] .= low_value
            sample_col[design_col .== 1] .= high_value
        end
    end
    return sample_matrix
end

function run_model(sample_matrix::AbstractArray, model)
    """

    param: sample_matrix
          A matrix where each row corresponds to a input vector for "model"
    param: model
          The function/model to be executed on each row of the sample_matrix.
          Must return a scalar.
    return:
          A vector containing the result of running each row of the
          sample_matrix through model.
    """

    samples = size(sample_matrix)[1]
    y_out = zeros(samples)
    Threads.@threads for i in 1:samples
        y_out[i] = model(@view sample_matrix[i, :])
    end
    return y_out
end

function ff_main_effects(design_matrix::AbstractArray, response_values::AbstractArray)
    """
         Computes the main effect for each parameter

         param: design_matrix

              The matrix outputted by 'ff_design' i.e a 2-level fractional
              factorial design matrix of -1s and 1s with a design of resolution IV.
              The number of rows is 2*number of parameters and the
              number of columns is the number of parameters.

         param: response_values

              An array of values corresponding to the result of running the
              model on each row of a sample matrix generated by the design_matrix

         return: main_effects

             A vector of the main effects of each parameter on the output.

          See Saltelli, A. equation 2.32 for details.
    """

    num_rows, num_cols = size(design_matrix)
    @assert num_rows==length(response_values) "Number of rows in design matrix must match number of responses"

    main_effects = zeros(num_cols)
    for column in 1:num_cols
        contrast_vector = @view design_matrix[:, column]
        main_effects[column] = dot(response_values, contrast_vector)
    end
    return main_effects ./ num_rows
end

function gsa(f, method::FractionalFactorial; num_params, p_range = nothing, kwargs...)
    design_matrix = generate_ff_design_matrix(num_params)
    sample_matrix = generate_ff_sample_matrix(design_matrix, p_range)

    response_vec = run_model(sample_matrix, f)

    main_effects = ff_main_effects(design_matrix, response_vec)
    return main_effects, main_effects .^ 2
end

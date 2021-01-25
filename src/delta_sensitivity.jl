struct DeltaSensitivity <: GSAMethod
    nboot::Int
    conf_level::Float64
    Ygrid_length::Int
    num_classes::Int
end


function _calc_delta(Xi, Y, Ygrid, class_cutoffs)

    # Make sure Ys are not identical, otherwise KDE will be undefined.
    # If Ys are identical, then we know X and Y are independent, so return 0
    if all(Y[1] .== Y)
            return 0
    end

    N = length(Y) # Number of simulations
    @assert length(Xi) == N # Length of Y should equal length of X

     # Model pdf of Y using KDE, kde uses normal kernel by default
    fy = pdf(kde(Y), Ygrid) # eq 23.1

    # Get probability of each y in Ygrid.
    x_rank = competerank(Xi) # Does what scipy.stats rankdata does. If tie, all tied values get same rank.
    d_hat = 0 # the delta estimator.

    # Iterate over each class
    weighted_class_seps = zeros(length(class_cutoffs)-1)
    for j in 1:length(class_cutoffs)-1

            # get X and Y indicies for samples that are in this class (where
            # class designation is based on the X value)
            condition(x) = (x > class_cutoffs[j]) ==  (x <= class_cutoffs[j+1])
            in_class_indices = findall(condition, x_rank)
            number_in_class = length(in_class_indices)

            # Get the the subset of Y values in this class
            in_class_Y = Y[in_class_indices]

            if length(in_class_Y) == 0
                    continue
            end

            # get the separation between the total y pdf and the condition y pdf induced by this class
            fyc = pdf(kde(in_class_Y), Ygrid) # eq 23.2 - Estimated conditional distribution of y (using normal kernel)
            pdf_diff = abs.(fy .- fyc) # eq 24

            # Use trapezoidal rule to estimate the difference between the curves.
            class_separation = trapz(Ygrid, pdf_diff) # eq 25

            # Increment estimator
            weighted_class_seps[j] = number_in_class * class_separation # eq 26
    end

    d_hat = sum(weighted_class_seps)/(2*N)
    return d_hat
end

function gsa()

    N = length(Y)

    # Create number of classes and class cutoffs.
    if num_classes == nothing
            exp = (2 / (7 + tanh((1500 - N) / 500)))
            M = Integer(round(min(Integer(ceil(N^exp)), 48))) # Number of classes
    else
            M = num_classes
    end
    class_cutoffs =  range(0, N, length=M+1) # class cutoffs.

    # quadrature points.
    # Length should be a power of 2 for efficient FFT in kernel density estimates.
    Ygrid = range(minimum(Y), maximum(Y), length=Ygrid_length)

    dims = size(X_matrix)
    if length(dims) == 2
            num_factors = dims[2]
    else
            num_factors = 1
    end

    deltas = zeros(num_factors)
    adjusted_deltas = zeros(num_factors)
    adjusted_deltas_conf = zeros(num_factors)
    adjusted_deltas_low = zeros(num_factors)
    adjusted_deltas_high = zeros(num_factors)

    Threads.@threads for factor_num in 1:num_factors
            Xi = view(X_matrix, :, factor_num)

            delta = _calc_delta(Xi, Y, Ygrid, class_cutoffs)
            deltas[factor_num] = delta

            # eq. 30, bias reduction via bootstrapping.
            bootstrap_deltas = zeros(num_resamples)
            r = rand(1:N, num_resamples, N)
            for i in 1:num_resamples
                    r_i = r[i, :]
                    bootstrap_deltas[i] = _calc_delta(Xi[r_i], Y[r_i], Ygrid, class_cutoffs)
            end

            adjusted_deltas[factor_num] = 2 * delta - mean(bootstrap_deltas)
            band = quantile(Normal(0.0, 1.0), 0.5+conf_level/2)*std(bootstrap_deltas)/(sqrt(num_resamples))
            adjusted_deltas_low[factor_num] = adjusted_deltas[factor_num] - band
            adjusted_deltas_high[factor_num] = adjusted_deltas[factor_num] + band
    end

    return deltas, adjusted_deltas, adjusted_deltas_low, adjusted_deltas_high
end
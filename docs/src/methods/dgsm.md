```julia
"""
The inputs for DGSM are as follows:
1.f: 
    This is the input function based on which the values of DGSM are to be evaluated
    Eg- f(x) = x[1]+x[2]^2
        This is function in 2 variables
2.samples:
    Depicts the number of sampling set of points to be used for evaluation of E(a), E(|a|) and E(a^2)
    a = partial derivative of f wrt x_i
3.distri:
    Array of distribution of respective variables
    Eg- dist = [Normal(5,6),Uniform(2,3)]
    for two variables
4.crossed:
    A string(True/False) which act as indicator for computation of DGSM crossed indices
    Eg- a True value over there will lead to evauation of crossed indices
"""
gsa(f, method::DGSM, distr::AbstractArray; samples::Int, kwargs...)
```

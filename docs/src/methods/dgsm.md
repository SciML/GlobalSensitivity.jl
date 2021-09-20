# Derivative based Global Sensitivity Measure Method

```julia
struct DGSM <: GSAMethod
    crossed::Bool 
end
```

The keyword arguments for DGSM are as follows:

- `crossed`: A string(True/False) which act as indicator for computation of DGSM crossed indices. Defaults to `false`.

## Method Details

The DGSM method takes a probability distribution for each of the 
parameters and samples are obtained from the distributions to create 
random parameter sets. Derivatives of the function being analysed are 
then computed at the sampled parameters and specific statistics of those 
derivatives are used. The paper by [Sobol and Kucherenko](http://www.sciencedirect.com/science/article/pii/S0378475409000354) 
discusses the relationship between the DGSM results, `tao` and 
`sigma` and the Morris elementary effects and Sobol Indices.

### API

```julia
function gsa(f, method::DGSM, dist::AbstractArray; samples::Int, kwargs...)
```

`dist`: Array of distribution of respective variables. E.g. `dist = [Normal(5,6),Uniform(2,3)]` for two variables.

### Example

```julia
using GlobalSensitivity, Test, Distributions

samples = 2000000

f1(x) = x[1] + 2*x[2] + 6.00*x[3]
dist1 = [Uniform(4,10),Normal(4,23),Beta(2,3)]
b =  gsa(f1,DGSM(),dist1,samples=samples)
```
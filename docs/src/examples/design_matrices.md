# Design Matrices

For the Sobol Method, we can have more control over the sampled points by generating design matrices.
Doing it in this manner lets us directly specify a quasi-Monte Carlo sampling method for the parameter space. Here
we use [QuasiMonteCarlo.jl](https://github.com/SciML/QuasiMonteCarlo.jl) to generate the design matrices
as follows:

```julia
using GlobalSensitivity, QuasiMonteCarlo, Plots
N = 10000
lb = [1.0, 1.0, 1.0, 1.0]
ub = [5.0, 5.0, 5.0, 5.0]
sampler = SobolSample()
A,B = QuasiMonteCarlo.generate_design_matrices(N,lb,ub,sampler)
```

and now we tell it to calculate the Sobol indices on these designs for the function `f1` we defined in the Lotka Volterra example:

```julia
sobol_result = gsa(f1,Sobol(),A,B)
```

We plot the first order and total order Sobol Indices for the parameters (`a` and `b`).

```julia

p1 = bar(["a","b","c","d"],sobol_result.ST[1,:],title="Total Order Indices prey",legend=false)
p2 = bar(["a","b","c","d"],sobol_result.S1[1,:],title="First Order Indices prey",legend=false)
p1_ = bar(["a","b","c","d"],sobol_result.ST[2,:],title="Total Order Indices predator",legend=false)
p2_ = bar(["a","b","c","d"],sobol_result.S1[2,:],title="First Order Indices predator",legend=false)
plot(p1,p2,p1_,p2_)
```
![sobolplot](../assets/sobolbars.png)

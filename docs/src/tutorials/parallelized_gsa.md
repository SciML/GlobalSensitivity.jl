# Parallelized Morris and Sobol Sensitivity Analysis of an ODE

Let's run GSA on the Lotka-Volterra model to and study the sensitivity of the maximum of predator population and the average prey population.

```julia
using GlobalSensitivity, Statistics, OrdinaryDiffEq #load packages
```

First, let's define our model:

```julia
function f(du,u,p,t)
  du[1] = p[1]*u[1] - p[2]*u[1]*u[2] #prey
  du[2] = -p[3]*u[2] + p[4]*u[1]*u[2] #predator
end
u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(f,u0,tspan,p)
t = collect(range(0, stop=10, length=200))
```

Now, let's create a function that takes in a parameter set and calculates the maximum of the predator population and the
average of the prey population for those parameter values. To do this, we will make use of the `remake` function, which
creates a new `ODEProblem`, and use the `p` keyword argument to set the new parameters:

```julia
f1 = function (p)
  prob1 = remake(prob;p=p)
  sol = solve(prob1,Tsit5();saveat=t)
  [mean(sol[1,:]), maximum(sol[2,:])]
end
```

Now, let's perform a Morris global sensitivity analysis on this model. We specify that the parameter range is
`[1,5]` for each of the parameters, and thus call:

```julia
m = gsa(f1,Morris(total_num_trajectory=1000,num_trajectory=150),[[1,5],[1,5],[1,5],[1,5]])
```
Let's get the means and variances from the `MorrisResult` struct.

```julia
m.means
2×2 Array{Float64,2}:
 0.474053  0.114922
 1.38542   5.26094

m.variances
2×2 Array{Float64,2}:
 0.208271    0.0317397
 3.07475   118.103
```

Let's plot the result

```julia
scatter(m.means[1,:], m.variances[1,:],series_annotations=[:a,:b,:c,:d],color=:gray)
scatter(m.means[2,:], m.variances[2,:],series_annotations=[:a,:b,:c,:d],color=:gray)
```

For the Sobol method, we can similarly do:

```julia
m = gsa(f1,Sobol(),[[1,5],[1,5],[1,5],[1,5]],N=1000)
```

## Direct Use of Design Matrices

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

![sobolbars](https://user-images.githubusercontent.com/23134958/127019349-686f968d-7c8a-4dc4-abdc-c70f58b043dd.png)

## Parallelizing the Global Sensitivity Analysis

In all of the previous examples, `f(p)` was calculated serially. However, we can parallelize our computations
by using the batch interface. In the batch interface, each column `p[:,i]` is a set of parameters, and we output
a column for each set of parameters. Here we showcase using the [Ensemble Interface](https://diffeq.sciml.ai/stable/features/ensemble/) to use
`EnsembleGPUArray` to perform automatic multithreaded-parallelization of the ODE solves.

```julia
using GlobalSensitivity, QuasiMonteCarlo, OrdinaryDiffEq

function f(du,u,p,t)
  du[1] = p[1]*u[1] - p[2]*u[1]*u[2] #prey
  du[2] = -p[3]*u[2] + p[4]*u[1]*u[2] #predator
end

u0 = [1.0;1.0]
tspan = (0.0,10.0)
p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(f,u0,tspan,p)
t = collect(range(0, stop=10, length=200))

f1 = function (p)
  prob_func(prob,i,repeat) = remake(prob;p=p[:,i])
  ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)
  sol = solve(ensemble_prob,Tsit5(),EnsembleThreads();saveat=t,trajectories=size(p,2))
  # Now sol[i] is the solution for the ith set of parameters
  out = zeros(2,size(p,2))
  for i in 1:size(p,2)
    out[1,i] = mean(sol[i][1,:])
    out[2,i] = maximum(sol[i][2,:])
  end
  out
end
```

And now to do the parallelized calls we simply add the `batch=true` keyword argument:

```julia
sobol_result = gsa(f1,Sobol(),A,B,batch=true)
```

This user-side parallelism thus allows you to take control, and thus for example you can use
[DiffEqGPU.jl](https://github.com/SciML/DiffEqGPU.jl) for automated GPU-parallelism of
the ODE-based global sensitivity analysis!

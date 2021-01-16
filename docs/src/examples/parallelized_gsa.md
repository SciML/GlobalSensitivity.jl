# Parallelized GSA Example

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

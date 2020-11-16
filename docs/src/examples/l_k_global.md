# Lotka-Volterra Global Sensitivities

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

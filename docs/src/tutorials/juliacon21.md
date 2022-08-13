# Global Sensitivity Analysis of the Lotka-Volterra model

The tutorial covers a workflow of using GlobalSensitivity.jl on the [Lotka-Volterra differential equations](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations).
We showcase how to use multiple GSA methods, analyse their results and leverage Julia's parallelism capabilities to
perform Global Sensitivity analysis at scale.

```@example lv
using GlobalSensitivity, QuasiMonteCarlo, OrdinaryDiffEq, Statistics, CairoMakie

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
    prob1 = remake(prob;p=p)
    sol = solve(prob1,Tsit5();saveat=t)
    return [mean(sol[1,:]), maximum(sol[2,:])]
end

bounds = [[1,5],[1,5],[1,5],[1,5]]

reg_sens = gsa(f1, RegressionGSA(true), bounds, samples = 200)
fig = Figure(resolution = (600, 400))
ax, hm = CairoMakie.heatmap(fig[1,1], reg_sens.partial_correlation, axis = (xticksvisible = false,yticksvisible = false, yticklabelsvisible = false, xticklabelsvisible = false, title = "Partial correlation"))
Colorbar(fig[1, 2], hm)
ax, hm = CairoMakie.heatmap(fig[2,1], reg_sens.standard_regression, axis = (xticksvisible = false,yticksvisible = false, yticklabelsvisible = false, xticklabelsvisible = false, title = "Standard regression"))
Colorbar(fig[2, 2], hm)
fig
```

![heatmapreg](https://user-images.githubusercontent.com/23134958/127019339-607b8d0b-6c38-4a18-b62e-e3ea0ae40941.png)

```@example lv
using StableRNGs
_rng = StableRNG(1234)
morris_sens = gsa(f1, Morris(), bounds, rng = _rng)
fig = Figure(resolution = (600, 400))
scatter(fig[1,1], [1,2,3,4], morris_sens.means_star[1,:], color = :green, axis = (xticksvisible = false, xticklabelsvisible = false, title = "Prey",))
scatter(fig[1,2], [1,2,3,4], morris_sens.means_star[2,:], color = :red, axis = (xticksvisible = false, xticklabelsvisible = false, title = "Predator",))
fig
```

![morrisscat](https://user-images.githubusercontent.com/23134958/127019346-2b5548c5-f4ec-4547-9f8f-af3e4b4c317c.png)

```@example lv
sobol_sens = gsa(f1, Sobol(), bounds, samples=500)
efast_sens = gsa(f1, eFAST(), bounds, samples=500)
fig = Figure(resolution = (600, 400))
barplot(fig[1,1], [1,2,3,4], sobol_sens.S1[1, :], color = :green, axis = (xticksvisible = false, xticklabelsvisible = false, title = "Prey (Sobol)", ylabel = "First order"))
barplot(fig[2,1], [1,2,3,4], sobol_sens.ST[1, :], color = :green, axis = (xticksvisible = false, xticklabelsvisible = false, ylabel = "Total order"))
barplot(fig[1,2], [1,2,3,4], efast_sens.S1[1, :], color = :red, axis = (xticksvisible = false, xticklabelsvisible = false, title = "Prey (eFAST)"))
barplot(fig[2,2], [1,2,3,4], efast_sens.ST[1, :], color = :red, axis = (xticksvisible = false, xticklabelsvisible = false))
fig

fig = Figure(resolution = (600, 400))
barplot(fig[1,1], [1,2,3,4], sobol_sens.S1[2, :], color = :green, axis = (xticksvisible = false, xticklabelsvisible = false, title = "Predator (Sobol)", ylabel = "First order"))
barplot(fig[2,1], [1,2,3,4], sobol_sens.ST[2, :], color = :green, axis = (xticksvisible = false, xticklabelsvisible = false, ylabel = "Total order"))
barplot(fig[1,2], [1,2,3,4], efast_sens.S1[2, :], color = :red, axis = (xticksvisible = false, xticklabelsvisible = false, title = "Predator (eFAST)"))
barplot(fig[2,2], [1,2,3,4], efast_sens.ST[2, :], color = :red, axis = (xticksvisible = false, xticklabelsvisible = false))
fig
```

![sobolefastprey](https://user-images.githubusercontent.com/23134958/127019361-8d625107-7f9c-44b5-a0dc-489bd512b7dc.png)
![sobolefastpred](https://user-images.githubusercontent.com/23134958/127019358-8bd0d918-e6fd-4929-96f1-d86330d91c69.png)

```@example lv
using QuasiMonteCarlo
samples = 500
lb = [1.0, 1.0, 1.0, 1.0]
ub = [5.0, 5.0, 5.0, 5.0]
sampler = SobolSample()
A,B = QuasiMonteCarlo.generate_design_matrices(samples,lb,ub,sampler)
sobol_sens_desmat = gsa(f1,Sobol(),A,B)


f_batch = function (p)
  prob_func(prob,i,repeat) = remake(prob;p=p[:,i])
  ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)

  sol = solve(ensemble_prob, Tsit5(), EnsembleThreads(); saveat=t, trajectories=size(p,2))

  out = zeros(2,size(p,2))

  for i in 1:size(p,2)
    out[1,i] = mean(sol[i][1,:])
    out[2,i] = maximum(sol[i][2,:])
  end

  return out
end

sobol_sens_batch = gsa(f_batch,Sobol(),A,B,batch=true)

@time gsa(f1,Sobol(),A,B)
@time gsa(f_batch,Sobol(),A,B,batch=true)
```

```@example lv
f1 = function (p)
           prob1 = remake(prob;p=p)
           sol = solve(prob1,Tsit5();saveat=t)
       end
sobol_sens = gsa(f1, Sobol(nboot = 20), bounds, samples=500)
fig = Figure(resolution = (600, 400))
ax, hm = CairoMakie.scatter(fig[1,1], sobol_sens.S1[1][1,2:end], label = "Prey", markersize = 4)
CairoMakie.scatter!(fig[1,1], sobol_sens.S1[1][2,2:end], label = "Predator", markersize = 4)

# Legend(fig[1,2], ax)

ax, hm = CairoMakie.scatter(fig[1,2], sobol_sens.S1[2][1,2:end], label = "Prey", markersize = 4)
CairoMakie.scatter!(fig[1,2], sobol_sens.S1[2][2,2:end], label = "Predator", markersize = 4)

ax, hm = CairoMakie.scatter(fig[2,1], sobol_sens.S1[3][1,2:end], label = "Prey", markersize = 4)
CairoMakie.scatter!(fig[2,1], sobol_sens.S1[3][2,2:end], label = "Predator", markersize = 4)

ax, hm = CairoMakie.scatter(fig[2,2], sobol_sens.S1[4][1,2:end], label = "Prey", markersize = 4)
CairoMakie.scatter!(fig[2,2], sobol_sens.S1[4][2,2:end], label = "Predator", markersize = 4)

title = Label(fig[0,:], "First order Sobol indices")
legend = Legend(fig[2,3], ax)
```

![timeseriessobollv](https://user-images.githubusercontent.com/23134958/156987652-85958bde-ae73-4f71-8555-318f779257ad.png)

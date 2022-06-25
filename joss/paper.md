---
title: 'GlobalSensitivity.jl: julia package for performant global sensitivity analysis'
tags:
  - julia
  - global sensitivity analysis
authors:
  - name: Vaibhav Kumar Dixit
    orcid: 0000-0001-7763-2717
    corresponding: true
    affiliation: 1
  - name: Christopher Rackauckas
    affiliation: "1, 2"
affiliations:
 - name: Julia Computing
   index: 1
 - name: MIT
   index: 2
date: 21 June 2022
bibliography: paper.bib

---

# Summary

Global Sensitivity Analysis (GSA) methods are used to quantify the uncertainty in
output of a model with respect to the parameters. These methods allow practitioners to
measure both parameter's individual contributions and the contribution of their interactions
to the output uncertainity. GlobalSensitivity.jl is a julia [@Bezanson2017] package containing implementation of some the most popular GSA methods. Currently it supports Delta Moment-Independent [@Borgonovo2007,@Plischke2013], DGSM [@Sobol2009], EASI [@Plischke2010] [@Plischke2012], eFAST [@Saltelli1999] [@Saltelli1998], Morris [@Morris1991] [@Campolongo2007], Fractional Factorial [@Saltelli2008b] , RBD-FAST [@Tarantola2006], Sobol [@Saltelli2008b] [@Sobol2001] [@Saltelli2002a] and Regression based sensitivity [@Guido2016] methods.

# Statement of need

Global Sensitivity Analysis has become an essential part of modeling workflow for practitioners in various fields such as Quantitative Systems Pharmacology and Environmental Modeling [@saltelli2020five] [@JAKEMAN2006602] [@sher2022quantitative] [@zhang2015sobol]. It can be used primarily in two stages, either before parameter estimation to simplify the fitting problem by fixing unimportant parameters or for analysis of the input parameters' influence on the output.

GlobalSensitivity.jl provides julia implementation of some of the popular GSA methods. Thus it benefits from the performance advantage of julia, provides a convenient unified API for different GSA methods by leveraging multiple dispatch and has a parallelized implementation for some of the methods.

This package allows users to conveniently perform GSA on arbitrary functions and get the sensitivity analysis results and at the same time provides out of the box support for differential equations based models defined using the SciML interface [@Rackauckas2017DifferentialEquationsjlA].

## Examples

The following examples cover a workflow of using GlobalSensitivity.jl on the Lotka-Volterra differential equation, popularly known as the predator-prey model. We showcase how to use multiple GSA methods, analyse their results and leverage Julia's parallelism capabilities to perform Global Sensitivity analysis at scale. The plots have been created using the Makie.jl package [@DanischKrumbiegel2021].

The function of interest, for performing GSA, is defined to be the mean of the prey population and maximum of the predator population.

First, we use the Regression based method and plot the partial correlation and standard regression coefficients as a heatmap.

```julia
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

reg_sens = gsa(f1, RegressionGSA(true), bounds)
fig = Figure(resolution = (600, 400))
ax, hm = CairoMakie.heatmap(fig[1,1], reg_sens.partial_correlation,
                            figure = (resolution = (300, 200),),
                            axis = (xticksvisible = false,
                            yticksvisible = false,
                            yticklabelsvisible = false,
                            xticklabelsvisible = false,
                            title = "Partial correlation"))
Colorbar(fig[1, 2], hm)
ax, hm = CairoMakie.heatmap(fig[2,1], reg_sens.standard_regression,
                            figure = (resolution = (300, 200),),
                            axis = (xticksvisible = false,
                            yticksvisible = false,
                            yticklabelsvisible = false,
                            xticklabelsvisible = false,
                            title = "Standard regression"))
Colorbar(fig[2, 2], hm)
fig
```

![heatmapreg](https://user-images.githubusercontent.com/23134958/127019339-607b8d0b-6c38-4a18-b62e-e3ea0ae40941.png)

Next, the Morris method is used and results are visualized as a scatter plot.

```julia
using StableRNGs
_rng = StableRNG(1234)
morris_sens = gsa(f1, Morris(), bounds, rng = _rng)
fig = Figure(resolution = (300, 200))
scatter(fig[1,1], [1,2,3,4], morris_sens.means_star[1,:],
        color = :green, axis = (xticksvisible = false,
        xticklabelsvisible = false, title = "Prey",))
scatter(fig[1,2], [1,2,3,4], morris_sens.means_star[2,:],
        color = :red, axis = (xticksvisible = false,
        xticklabelsvisible = false, title = "Predator",))
fig
```

![morrisscat](https://user-images.githubusercontent.com/23134958/127019346-2b5548c5-f4ec-4547-9f8f-af3e4b4c317c.png)

Here we show use of the Sobol and eFAST methods, the first order and total order indices are plotted for both the dependent variables for all four parameters.

```julia
sobol_sens = gsa(f1, Sobol(), bounds, N=5000)
efast_sens = gsa(f1, eFAST(), bounds)
fig = Figure(resolution = (300, 200))
barplot(fig[1,1], [1,2,3,4], sobol_sens.S1[1, :],
        color = :green, axis = (xticksvisible = false,
        xticklabelsvisible = false, title = "Prey (Sobol)",
        ylabel = "First order"))
barplot(fig[2,1], [1,2,3,4], sobol_sens.ST[1, :],
        color = :green, axis = (xticksvisible = false,
        xticklabelsvisible = false, ylabel = "Total order"))
barplot(fig[1,2], [1,2,3,4], efast_sens.S1[1, :],
        color = :red, axis = (xticksvisible = false,
        xticklabelsvisible = false, title = "Prey (eFAST)"))
barplot(fig[2,2], [1,2,3,4], efast_sens.ST[1, :],
        color = :red, axis = (xticksvisible = false,
        xticklabelsvisible = false))
fig

fig = Figure(resolution = (300, 200))
barplot(fig[1,1], [1,2,3,4], sobol_sens.S1[2, :],
        color = :green, axis = (xticksvisible = false,
        xticklabelsvisible = false, title = "Predator (Sobol)",
        ylabel = "First order"))
barplot(fig[2,1], [1,2,3,4], sobol_sens.ST[2, :],
        color = :green, axis = (xticksvisible = false,
        xticklabelsvisible = false, ylabel = "Total order"))
barplot(fig[1,2], [1,2,3,4], efast_sens.S1[2, :],
        color = :red, axis = (xticksvisible = false,
        xticklabelsvisible = false, title = "Predator (eFAST)"))
barplot(fig[2,2], [1,2,3,4], efast_sens.ST[2, :],
        color = :red, axis = (xticksvisible = false,
        xticklabelsvisible = false))
fig
```

![sobolefastprey](https://user-images.githubusercontent.com/23134958/127019361-8d625107-7f9c-44b5-a0dc-489bd512b7dc.png)
![sobolefastpred](https://user-images.githubusercontent.com/23134958/127019358-8bd0d918-e6fd-4929-96f1-d86330d91c69.png)

Leveraging the batch interface it is possible to parallelize the Sobol indices calculation, this is showcased in the example below.

```julia
using QuasiMonteCarlo
N = 5000
lb = [1.0, 1.0, 1.0, 1.0]
ub = [5.0, 5.0, 5.0, 5.0]
sampler = SobolSample()
A,B = QuasiMonteCarlo.generate_design_matrices(N,lb,ub,sampler)
sobol_sens_desmat = gsa(f1,Sobol(),A,B)


f_batch = function (p)
  prob_func(prob,i,repeat) = remake(prob;p=p[:,i])
  ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)

  sol = solve(ensemble_prob, Tsit5(), EnsembleThreads();
              saveat=t, trajectories=size(p,2))

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

As mentioned before, you can call the `gsa` function directly on the differential equation solution and compute sensitivities across the timeseries. This is demonstrated in the example below, the Sobol indices for each time point are then displayed as a plot.

```julia
f1 = function (p)
           prob1 = remake(prob;p=p)
           sol = solve(prob1,Tsit5();saveat=t)
       end
sobol_sens = gsa(f1, Sobol(nboot = 20), bounds, N=5000)
fig = Figure(resolution = (300, 200))
ax, hm = CairoMakie.scatter(fig[1,1], sobol_sens.S1[1][1,2:end],
                            label = "Prey", markersize = 4)
CairoMakie.scatter!(fig[1,1], sobol_sens.S1[1][2,2:end],
                    label = "Predator", markersize = 4)

# Legend(fig[1,2], ax)

ax, hm = CairoMakie.scatter(fig[1,2], sobol_sens.S1[2][1,2:end],
                            label = "Prey", markersize = 4)
CairoMakie.scatter!(fig[1,2], sobol_sens.S1[2][2,2:end],
                    label = "Predator", markersize = 4)

ax, hm = CairoMakie.scatter(fig[2,1], sobol_sens.S1[3][1,2:end],
                            label = "Prey", markersize = 4)
CairoMakie.scatter!(fig[2,1], sobol_sens.S1[3][2,2:end],
                    label = "Predator", markersize = 4)

ax, hm = CairoMakie.scatter(fig[2,2], sobol_sens.S1[4][1,2:end],
                            label = "Prey", markersize = 4)
CairoMakie.scatter!(fig[2,2], sobol_sens.S1[4][2,2:end],
                    label = "Predator", markersize = 4)

title = Label(fig[0,:], "First order Sobol indices")
legend = Legend(fig[2,3], ax)
```

![timeseriessobollv](https://user-images.githubusercontent.com/23134958/156987652-85958bde-ae73-4f71-8555-318f779257ad.png)

# References

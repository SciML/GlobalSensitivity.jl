# GlobalSensitivity.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/GlobalSensitivity/stable/)

[![codecov](https://codecov.io/gh/SciML/GlobalSensitivity.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SciML/GlobalSensitivity.jl)
[![Build Status](https://github.com/SciML/GlobalSensitivity.jl/workflows/CI/badge.svg)](https://github.com/SciML/GlobalSensitivity.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04561/status.svg)](https://doi.org/10.21105/joss.04561)

GlobalSensitivity.jl package contains implementation of some of the most popular GSA methods. Currently it supports Delta Moment-Independent, DGSM, EASI, eFAST, Morris, Mutual Information, Fractional Factorial, RBD-FAST, RSA, Sobol and Regression based sensitivity methods.

## Tutorials and Documentation

For information on using the package,
[see the stable documentation](https://docs.sciml.ai/GlobalSensitivity/stable/). Use the
[in-development documentation](https://docs.sciml.ai/GlobalSensitivity/dev/) for the version of
the documentation, which contains the unreleased features.

## Installation

The GlobalSensitivity.jl package can be installed with julia's package manager as shown below:

```julia
using Pkg
Pkg.add("GlobalSensitivity")
```

## General Interface

The general interface for performing global sensitivity analysis using this package is:

```julia
res = gsa(f, method, param_range; samples, batch = false)
```

## Example

### Sobol method on the [Ishigami function](https://www.sfu.ca/%7Essurjano/ishigami.html).

Serial execution

```julia
function ishi(X)
    A = 7
    B = 0.1
    sin(X[1]) + A * sin(X[2])^2 + B * X[3]^4 * sin(X[1])
end

n = 600000
lb = -ones(4) * π
ub = ones(4) * π
sampler = SobolSample()
A, B = QuasiMonteCarlo.generate_design_matrices(n, lb, ub, sampler)

res1 = gsa(ishi, Sobol(order = [0, 1, 2]), A, B)
```

Using batching interface

```julia
function ishi_batch(X)
    A = 7
    B = 0.1
    @. sin(X[1, :]) + A * sin(X[2, :])^2 + B * X[3, :]^4 * sin(X[1, :])
end

res2 = gsa(ishi_batch, Sobol(), A, B, batch = true)
```

### Regression based and Morris method sensitivity analysis of Lotka Volterra model.

```julia
using GlobalSensitivity, QuasiMonteCarlo, OrdinaryDiffEq, Statistics, CairoMakie

function f(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2] #prey
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2] #predator
end

u0 = [1.0; 1.0]
tspan = (0.0, 10.0)
p = [1.5, 1.0, 3.0, 1.0]
prob = ODEProblem(f, u0, tspan, p)
t = collect(range(0, stop = 10, length = 200))

f1 = function (p)
    prob1 = remake(prob; p = p)
    sol = solve(prob1, Tsit5(); saveat = t)
    return [mean(sol[1, :]), maximum(sol[2, :])]
end

bounds = [[1, 5], [1, 5], [1, 5], [1, 5]]

reg_sens = gsa(f1, RegressionGSA(true), bounds)
fig = Figure(resolution = (600, 400))
ax,
hm = CairoMakie.heatmap(fig[1, 1], reg_sens.partial_correlation,
    figure = (resolution = (300, 200),),
    axis = (xticksvisible = false,
        yticksvisible = false,
        yticklabelsvisible = false,
        xticklabelsvisible = false,
        title = "Partial correlation"))
Colorbar(fig[1, 2], hm)
ax,
hm = CairoMakie.heatmap(fig[2, 1], reg_sens.standard_regression,
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

```julia
using StableRNGs
_rng = StableRNG(1234)
morris_sens = gsa(f1, Morris(), bounds, rng = _rng)
fig = Figure(resolution = (300, 200))
scatter(fig[1, 1], [1, 2, 3, 4], morris_sens.means_star[1, :],
    color = :green, axis = (xticksvisible = false,
        xticklabelsvisible = false, title = "Prey (Morris)"))
scatter(fig[1, 2], [1, 2, 3, 4], morris_sens.means_star[2, :],
    color = :red, axis = (xticksvisible = false,
        xticklabelsvisible = false, title = "Predator (Morris)"))
fig
```

![morrisscat](https://user-images.githubusercontent.com/23134958/127019346-2b5548c5-f4ec-4547-9f8f-af3e4b4c317c.png)

## Citing

If you use this software in your work, please cite:

```bib
@article{dixit2022globalsensitivity,
  title={GlobalSensitivity. jl: Performant and Parallel Global Sensitivity Analysis with Julia},
  author={Dixit, Vaibhav Kumar and Rackauckas, Christopher},
  journal={Journal of Open Source Software},
  volume={7},
  number={76},
  pages={4561},
  year={2022}
}
```

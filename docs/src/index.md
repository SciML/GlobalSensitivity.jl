# GlobalSensitivity.jl: Robust, Fast, and Parallel Global Sensitivity Analysis (GSA) in Julia

Global Sensitivity Analysis (GSA) methods are used to quantify the uncertainty in
output of a model with respect to the parameters. These methods allow practitioners to
measure both parameter's individual contributions and the contribution of their interactions
to the output uncertainity.

## Installation

To use this functionality, you must install GlobalSensitivity.jl:

```julia
]add GlobalSensitivity
using GlobalSensitivity
```

Note: GlobalSensitivity.jl is unrelated to the GlobalSensitivityAnalysis.jl package.

## General Interface

The general interface for performing global sensitivity analysis using this package is:

```@docs
gsa(f, method::GlobalSensitivity.GSAMethod, param_range; samples, batch = false)
```

The descriptions of the available methods can be found in the Methods section.
The `gsa` interface allows for utilizing batched functions with the `batch` kwarg discussed above for parallel
computation of GSA results.

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

## Reproducibility
```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```
```@example
using Pkg # hide
Pkg.status() # hide
```
```@raw html
</details>
```
```@raw html
<details><summary>and using this machine and Julia version.</summary>
```
```@example
using InteractiveUtils # hide
versioninfo() # hide
```
```@raw html
</details>
```
```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```
```@example
using Pkg # hide
Pkg.status(;mode = PKGMODE_MANIFEST) # hide
```
```@raw html
</details>
```
```@raw html
You can also download the 
<a href="
```
```@eval
using TOML
version = TOML.parse(read("../../Project.toml",String))["version"]
name = TOML.parse(read("../../Project.toml",String))["name"]
link = "https://github.com/SciML/"*name*".jl/tree/gh-pages/v"*version*"/assets/Manifest.toml"
```
```@raw html
">manifest</a> file and the
<a href="
```
```@eval
using TOML
version = TOML.parse(read("../../Project.toml",String))["version"]
name = TOML.parse(read("../../Project.toml",String))["name"]
link = "https://github.com/SciML/"*name*".jl/tree/gh-pages/v"*version*"/assets/Project.toml"
```
```@raw html
">project</a> file.
```
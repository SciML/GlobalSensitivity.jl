using Documenter, GlobalSensitivity

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

ENV["GKSwstype"] = "100"
using Plots, CairoMakie

include("pages.jl")

makedocs(sitename = "GlobalSensitivity.jl",
    authors = "Vaibhav Kumar Dixit",
    modules = [GlobalSensitivity],
    clean = true, doctest = false, linkcheck = true,
    warnonly = [:missing_docs],
    format = Documenter.HTML(assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/GlobalSensitivity/stable/"),
    pages = pages)

deploydocs(repo = "github.com/SciML/GlobalSensitivity.jl";
    push_preview = true)

using Documenter, GlobalSensitivity

ENV["GKSwstype"] = "100"
using Plots, CairoMakie

include("pages.jl")

makedocs(sitename = "GlobalSensitivity.jl",
         authors = "Vaibhav Kumar Dixit",
         clean = true,
         doctest = false,
         modules = [GlobalSensitivity],
         strict = [
             :doctest,
             :linkcheck,
             :parse_error,
             :example_block,
             # Other available options are
             # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :missing_docs, :setup_block
         ],
         format = Documenter.HTML(analytics = "UA-90474609-3",
                                  assets = ["assets/favicon.ico"],
                                  canonical = "https://globalsensitivity.sciml.ai/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/GlobalSensitivity.jl";
           push_preview = true)

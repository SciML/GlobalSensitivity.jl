using Documenter, GlobalSensitivity

include("pages.jl")

makedocs(sitename = "GlobalSensitivity.jl",
         authors = "Vaibhav Kumar Dixit",
         clean = true,
         doctest = false,
         modules = [GlobalSensitivity],
         format = Documenter.HTML(analytics = "UA-90474609-3",
                                  assets = ["assets/favicon.ico"],
                                  canonical = "https://globalsensitivity.sciml.ai/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/GlobalSensitivity.jl";
           push_preview = true)

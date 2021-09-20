using Documenter, GlobalSensitivity

makedocs(
    sitename="GlobalSensitivity.jl",
    authors="Vaibhav Kumar Dixit",
    clean=true,
    doctest=false,
    modules=[GlobalSensitivity],

    format=Documenter.HTML(assets=["assets/favicon.ico"],
                           canonical="https://globalsensitivity.sciml.ai/stable/"),

    pages=[
        "GlobalSensitivity.jl: Global Sensitivity Analysis (GSA)" => "index.md",
        "Tutorials" => Any[
          "tutorials/parallelized_gsa.md",
          "tutorials/juliacon21.md"
        ],
        "Methods" => Any[
          "methods/morris.md",
          "methods/sobol.md",
          "methods/regression.md",
          "methods/efast.md",
          "methods/delta.md",
          "methods/easi.md",
          "methods/fractional.md",
          "methods/rbdfast.md"
        ],
    ]
)

deploydocs(
    repo="github.com/SciML/GlobalSensitivity.jl";
    push_preview=true
)

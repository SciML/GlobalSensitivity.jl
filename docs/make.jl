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
        "Examples" => Any[
          "examples/l_k_global.md",
          "examples/design_matrices.md",
          "examples/parallelized_gsa.md"
        ],
        "Methods" => Any[
          "methods/morris.md",
          "methods/sobol.md",
          "methods/regression.md",
          "methods/efast.md"
        ],
    ]
)

deploydocs(
    repo="github.com/SciML/GlobalSensitivity.jl";
    push_preview=true
)

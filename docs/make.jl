using UnfoldDecode
using Documenter

DocMeta.setdocmeta!(UnfoldDecode, :DocTestSetup, :(using UnfoldDecode); recursive=true)

makedocs(;
    modules=[UnfoldDecode],
    authors="Benedikt V. Ehinger",
    repo="https://github.com/behinger/UnfoldDecode.jl/blob/{commit}{path}#{line}",
    sitename="UnfoldDecode.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://behinger.github.io/UnfoldDecode.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/behinger/UnfoldDecode.jl",
    devbranch="main",
)

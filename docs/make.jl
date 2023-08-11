using UnfoldDecode
using Documenter
using Glob
using Literate


GENERATED = joinpath(@__DIR__, "src", "literate")
for subfolder ∈ ["explanations","HowTo","tutorials","reference"]
    local SOURCE_FILES = Glob.glob(subfolder*"/*.jl", GENERATED)
    #config=Dict(:repo_root_path=>"https://github.com/unfoldtoolbox/UnfoldSim")
    foreach(fn -> Literate.markdown(fn, GENERATED*"/"*subfolder), SOURCE_FILES)

end

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
        "tutorials"=>[
            "Overlap corrected LDA" => "literate/tutorials/overlapcorrectedLDA.jl"
        ]
    ],
)

deploydocs(;
    repo="github.com/behinger/UnfoldDecode.jl",
    devbranch="main",
)

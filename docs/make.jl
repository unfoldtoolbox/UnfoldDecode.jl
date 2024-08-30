using UnfoldDecode
using Documenter
using Glob
using Literate


GENERATED = joinpath(@__DIR__, "src", "literate")
for subfolder ∈ ["explanations","HowTo","tutorials","BackToBack","reference"]
    local SOURCE_FILES = Glob.glob(subfolder*"/*.jl", GENERATED)
    #config=Dict(:repo_root_path=>"https://github.com/unfoldtoolbox/UnfoldSim")
    foreach(fn -> Literate.markdown(fn, GENERATED*"/"*subfolder), SOURCE_FILES)

end

DocMeta.setdocmeta!(UnfoldDecode, :DocTestSetup, :(using UnfoldDecode); recursive=true)

makedocs(;
    # modules=[UnfoldDecode],
    # authors="Benedikt V. Ehinger",
    authors="CXC2001",
    # repo="https://github.com/unfoldtoolbox/UnfoldDecode.jl/blob/{commit}{path}#{line}",
    # repo="https://github.com/CXC2001/UnfoldDecode.jl/blob/{commit}{path}#{line}",
    repo=Documenter.Remotes.GitHub("CXC2001/UnfoldDecode.jl"),
    sitename="UnfoldDecode.jl",
    # format=Documenter.HTML(;
    #     prettyurls=get(ENV, "CI", "false") == "true",
    #     # canonical="https://behinger.github.io/UnfoldDecode.jl",
    #     edit_link="main",
    #     assets=String[],
    # ),
    pages=[
        "Home" => "index.md",
        "tutorials"=>[
            "Overlap corrected LDA" => "literate/tutorials/overlapcorrectedLDA.md"
        ],
        "HowTo" =>[
            "Overlap corrected, multi-event, many options, LDA" => "literate/HowTo/overlapcorrectedLDA_options.md",
        ],
        "BackToBack" =>[
            "Quick Start b2b" => "literate/BackToBack/Quick_Start_b2b.md",
            "About Back-to-Back Decoding" => "literate/BackToBack/About_BacktoBack.md",
            "About b2b" => "literate/BackToBack/About_b2b.md",  
        ],
    ],
)

deploydocs(;
    # repo="github.com/unfoldtoolbox/UnfoldDecode.jl",
    repo="github.com/CXC2001/UnfoldDecode.jl",
    devbranch="main",
)

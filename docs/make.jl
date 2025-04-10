using UnfoldDecode
using Documenter
using Glob
using Literate



GENERATED = joinpath(@__DIR__, "src", "generated")
SOURCE = joinpath(@__DIR__, "literate")
for subfolder ∈ ["explanations", "HowTo", "tutorials", "reference"]
    local SOURCE_FILES = Glob.glob(subfolder * "/*.jl", SOURCE)
    foreach(fn -> Literate.markdown(fn, GENERATED * "/" * subfolder), SOURCE_FILES)

end


DocMeta.setdocmeta!(UnfoldDecode, :DocTestSetup, :(using UnfoldDecode); recursive = true)

makedocs(;
    modules = [UnfoldDecode],
    # authors="Benedikt V. Ehinger",
    authors = "Benedikt V. Ehinger, CXC2001",
    # repo="https://github.com/unfoldtoolbox/UnfoldDecode.jl/blob/{commit}{path}#{line}",
    # repo="https://github.com/CXC2001/UnfoldDecode.jl/blob/{commit}{path}#{line}",
    # repo=Documenter.Remotes.GitHub("CXC2001/UnfoldDecode.jl"),
    sitename = "UnfoldDecode.jl",
    warnonly = true,
    format = Documenter.HTML(;
        # prettyurls=get(ENV, "CI", "false") == "true",
        prettyurls = get(ENV, "CI", nothing) == "true",
        # canonical="https://behinger.github.io/UnfoldDecode.jl",
        edit_link = "main",
        assets = String[],
    ),
    pagesonly = true, # render only what is specified below
    pages = [
        "Home" => "index.md",
        "tutorials" =>
            ["Overlap corrected LDA" => "generated/tutorials/overlapcorrectedLDA.md"],
        "HowTo" => [
            "Overlap corrected, multi-event, many options, LDA" => "generated/HowTo/overlapcorrectedLDA_options.md",
        ],
        "BackToBack" => [
            "Tutorials" =>
                ["Quick Start b2b" => "generated/tutorials/b2b_quickstart.md"],
            "Explanations" => ["Why Back2Back" => "generated/explanations/b2b_why.md"],
            "Reference" =>
                ["Algorithm choice" => "generated/reference/b2b_algorithms.md"],
        ],
    ],
)

deploydocs(; repo = "github.com/unfoldtoolbox/UnfoldDecode.jl.git", devbranch = "main")

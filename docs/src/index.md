```@meta
CurrentModule = UnfoldDecode
```

# UnfoldDecode.jl Documentation

Welcome to [UnfoldDecode.jl](https://github.com/unfoldtoolbox/UnfoldDecode.jl)

```@raw html
<div style="width:60%; margin: auto;">
</div>
```

<!-- Documentation for [UnfoldDecode](https://github.com/behinger/UnfoldDecode.jl). -->
Documentation for [UnfoldDecode](https://github.com/CXC2001/UnfoldDecode.jl).

In UnfoldDecode.jl we develop new approaches, collect other algorithms, and provide tutorials, to decode overlapping and/or covariate-heavy EEG data.

## Key features

This toolbox right now implements:

- Overlap-corrected decoding ala [Gal Vishne, Leon Deouell et al.](https://doi.org/10.1101/2023.06.28.546397)
- Covariate-corrected decoding ala [back-to-back regression, Jean-Remy King et al.](https://doi.org/10.1016/j.neuroimage.2020.117028)

## Installation

```julia-repl
julia> using Pkg; Pkg.add("UnfoldDecode")
```

For more detailed instructions please refer to [Installing Julia & Unfold Packages](https://unfoldtoolbox.github.io/UnfoldDocs/main/installation/).

## Usage example

## Where to start: Learning roadmap

### First steps B2B

📌 Goal: Learn why to use back-to-back regression, next learn how easy it is to apply it to your data /
🔗 [B2B explained](@ref explainer-b2b), [B2B quickstart](@ref quickstart-b2b)

### First Steps Overlap correction

📌 Goal: Learn how to run overlap corrected Unfold-style deconvolution models /
🔗 [overlap-corrected decoding tutorial](@ref tutorial-overlap-decoding)

## Statement of need

```@raw html
<!---
Note: The statement of need is also used in the `README.md`. Make sure that they are synchronized.
-->
```

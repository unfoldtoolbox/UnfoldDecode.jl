# UnfoldDecode
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://behinger.github.io/UnfoldDecode.jl/dev/)
[![Build Status](https://github.com/behinger/UnfoldDecode.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/behinger/UnfoldDecode.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/behinger/UnfoldDecode.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/behinger/UnfoldDecode.jl)

Beta-stage toolbox to decode ERPs with overlap, e.g. from eye-tracking experiments.



Currently only the overlap corrected LDA¹ proposed by [Gal Vishne, Leon Deouell et al.](https://doi.org/10.1101/2023.06.28.546397) is implemented, but more to follow.

¹ actually any MLJ supported classification/regressoin model is already supported

## Quickstart

```julia
LDA = @load LDA pkg=MultivariateStats

des = Dict("fixation" => (@formula(0~1+condition+continuous),firbasis((-0.1,1.),100)));
uf_lda = fit(UnfoldDecodingModel,des,evt,dat,LDA(),"fixation"=>:condition)
```

Does the trick - you should probably do an Unfold.jl tutorial first though!
## Installation
Not yet registered thus you have to do:
```julia
using Pkg
Pkg.add(url="https://github.com/behinger/UnfoldDecode.jl")
using UnfoldDecode
```
once it is registered, this will simplify to `Pkg.add("UnfoldDecode")`
## Loading Data
have a look at PyMNE.jl to read the data. You need a data-matrix + DataFrames.jl event table (similar to EEGlabs EEG.events)

## Limitations
- Not thoroughly tested, no unit-tests yet!
- Missing features: e.g. No time generalization is available, but straight forward to implement with the current tooling.

## Citing

If you use this code, please cite this code + the appropriate paper/algorithm

## Contributions

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->



This project follows the [all-contributors](https://allcontributors.org/docs/en/specification) specification. 

Contributions of any kind welcome!
You can find the emoji key for the contributors [here](https://github.com/unfoldtoolbox/Unfold.jl/blob/main/docs/contrib-emoji.md).

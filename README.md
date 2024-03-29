# UnfoldDecode
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://behinger.github.io/UnfoldDecode.jl/dev/)
[![Build Status](https://github.com/behinger/UnfoldDecode.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/behinger/UnfoldDecode.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/behinger/UnfoldDecode.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/behinger/UnfoldDecode.jl)

Beta-stage toolbox to decode ERPs with overlap, e.g. from eye-tracking experiments.

> [!WARNING]
> No unit-tests implemented as of 2024-01-09 - use at your own risk!

Currently the following algorithms are implemented:

- [back-to-back regession](https://doi.org/10.1016/j.neuroimage.2020.117028) (`solver_b2b`, [tutorial how to use](https://unfoldtoolbox.github.io/Unfold.jl/dev/HowTo/custom_solvers/#Back2Back-regression)) 
- overlap corrected LDA¹ proposed by [Gal Vishne, Leon Deouell et al.](https://doi.org/10.1101/2023.06.28.546397) is implemented, but more to follow.

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

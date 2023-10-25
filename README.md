# [![logo_UnfoldDecode jl_120px](https://github.com/unfoldtoolbox/UnfoldDecode.jl/assets/57703446/965b93aa-33e1-420e-a707-1fe8d7e3bcbe)](https://github.com/unfoldtoolbox/UnfoldDecode.jl/tree/main)

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://behinger.github.io/UnfoldDecode.jl/dev/)
[![Build Status](https://github.com/behinger/UnfoldDecode.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/behinger/UnfoldDecode.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/behinger/UnfoldDecode.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/behinger/UnfoldDecode.jl)

|rERP|EEG visualisation|EEG Simulations|BIDS pipeline|Decode EEG data|Statistical testing|
|---|---|---|---|---|---|
| <a href="https://github.com/unfoldtoolbox/Unfold.jl/tree/main"><img src="https://github-production-user-asset-6210df.s3.amazonaws.com/10183650/277623787-757575d0-aeb9-4d94-a5f8-832f13dcd2dd.png"></a> | <a href="https://github.com/unfoldtoolbox/UnfoldMakie.jl"><img  src="https://github-production-user-asset-6210df.s3.amazonaws.com/10183650/277623793-37af35a0-c99c-4374-827b-40fc37de7c2b.png"></a>|<a href="https://github.com/unfoldtoolbox/UnfoldSim.jl"><img src="https://github-production-user-asset-6210df.s3.amazonaws.com/10183650/277623795-328a4ccd-8860-4b13-9fb6-64d3df9e2091.png"></a>|<a href="https://github.com/unfoldtoolbox/UnfoldBIDS.jl"><img src="https://github-production-user-asset-6210df.s3.amazonaws.com/10183650/277622460-2956ca20-9c48-4066-9e50-c5d25c50f0d1.png"></a>|<a href="https://github.com/unfoldtoolbox/UnfoldDecode.jl"><img src="https://github-production-user-asset-6210df.s3.amazonaws.com/10183650/277622487-802002c0-a1f2-4236-9123-562684d39dcf.png"></a>|<a href="https://github.com/unfoldtoolbox/UnfoldStats.jl"><img  src="https://github-production-user-asset-6210df.s3.amazonaws.com/10183650/277623799-4c8f2b5a-ea84-4ee3-82f9-01ef05b4f4c6.png"></a>|

Beta-stage toolbox to decode ERPs with overlap, e.g. from eye-tracking experiments.

Currently only the overlap corrected LDA¹ proposed by [Gal Vishne, Leon Deouell et al.](https://doi.org/10.1101/2023.06.28.546397) is implemented, but more to follow.

¹ actually any MLJ supported classification/regressoin model is already supported

## Install

### Julia
<details>
<summary>Click to expand</summary>

The recommended way to install julia is [juliaup](https://github.com/JuliaLang/juliaup).
It allows you to, e.g., easily update Julia at a later point, but also test out alpha/beta versions etc.

TL:DR; If you dont want to read the explicit instructions, just copy the following command

#### Windows

AppStore -> JuliaUp,  or `winget install julia -s msstore` in CMD

#### Mac & Linux

`curl -fsSL https://install.julialang.org | sh` in any shell
</details>

### UnfoldDecode

Not yet registered thus you have to do:
```julia
using Pkg
Pkg.add(url="https://github.com/behinger/UnfoldDecode.jl")
using UnfoldDecode
```
once it is registered, this will simplify to `Pkg.add("UnfoldDecode")`

## Quickstart

```julia
LDA = @load LDA pkg=MultivariateStats

des = Dict("fixation" => (@formula(0~1+condition+continuous),firbasis((-0.1,1.),100)));
uf_lda = fit(UnfoldDecodingModel,des,evt,dat,LDA(),"fixation"=>:condition)
```

Does the trick - you should probably do an Unfold.jl tutorial first though!

## Loading Data
have a look at PyMNE.jl to read the data. You need a data-matrix + DataFrames.jl event table (similar to EEGlabs EEG.events)

## Limitations
- Not thoroughly tested, no unit-tests yet!
- Missing features: e.g. No time generalization is available, but straight forward to implement with the current tooling.

## Contributions

Contributions are very welcome. These could be typos, bugreports, feature-requests, speed-optimization, new solvers, better code, better documentation.

### How-to Contribute

You are very welcome to raise issues and start pull requests!

### Adding Documentation

1. We recommend to write a Literate.jl document and place it in `docs/_literate/FOLDER/FILENAME.jl` with `FOLDER` being `HowTo`, `Explanation`, `Tutorial` or `Reference` ([recommended reading on the 4 categories](https://documentation.divio.com/)).
2. Literate.jl converts the `.jl` file to a `.md` automatically and places it in `doc/src/_literate/FILENAME.jl`.
3. Edit [make.jl](https://github.com/unfoldtoolbox/Unfold.jl/blob/main/docs/make.jl) with a reference to `doc/src/_literate/FILENAME.jl`

## Contributors (alphabetically)

- **Hannes Bonasch**
- **Benedikt Ehinger**


## Citation

If you use this code, please cite this code + the appropriate paper/algorithm

## Acknowledgements

Funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany´s Excellence Strategy – EXC 2075 – 390740016

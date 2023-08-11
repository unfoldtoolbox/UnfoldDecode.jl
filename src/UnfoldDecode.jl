module UnfoldDecode

using Unfold
import Unfold.coeftable
using MLJ
using MultivariateStats
using MLJBase
using DataFrames


LDA = @load LDA pkg=MultivariateStats


# Write your package code here.
include("types.jl")
include("decoding.jl")
include("fit.jl")
include("helper.jl")
include("overlap_corrected.jl")

export UnfoldDecodingModel
export coeftable
end

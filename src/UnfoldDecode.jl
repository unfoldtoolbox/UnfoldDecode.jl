module UnfoldDecode

using Unfold
import Unfold.coeftable
import Unfold.fit
using MLJ
using MultivariateStats
import MLJBase
using DataFrames
using Logging # to deactivate some MLJ output

# Write your package code here.
include("types.jl")
include("decoding.jl")
include("fit.jl")
include("helper.jl")
include("overlap_corrected.jl")
include("b2b.jl")

export UnfoldDecodingModel
export coeftable
export fit
export solver_b2b

end

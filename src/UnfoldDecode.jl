module UnfoldDecode

using Unfold
import Unfold.coeftable
import Unfold.fit
using Unfold: @maybe_threads
using MLJ
using MultivariateStats
#import MLJBase
using MLBase
using DataFrames
using ProgressMeter
using LinearAlgebra
using Logging # to deactivate some MLJ output

using MLJLinearModels, Tables
# using MLJScikitLearnInterface
using MultivariateStats
using MLJMultivariateStatsInterface
# using LIBSVM
# using MLJLIBSVMInterface
using XGBoost

# Write your package code here.
include("types.jl")
include("decoding.jl")
include("fit.jl")
include("helper.jl")
include("overlap_corrected.jl")
include("b2b/b2b.jl")
include("b2b/solvers.jl")

export UnfoldDecodingModel
export coeftable
export fit
export solver_b2b

end

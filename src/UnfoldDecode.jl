module UnfoldDecode

using Unfold
import Unfold.coeftable
import Unfold.fit
using MLJ
using MultivariateStats
import MLJBase
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
using DecisionTree
using MLJDecisionTreeInterface

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

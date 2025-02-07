using Unfold
using UnfoldMakie, CairoMakie
using UnfoldSim
using UnfoldDecode
using DataFrames
using Statistics
# using MLBase
# using MLJ
# using MLJLinearModels,Tables,

dat, evts = UnfoldSim.predef_eeg(; noiselevel = 0.1, return_epoched = true)

evts = rename(evts, :condition => :animal) # rename the column "condition" to "animal"
evts = rename(evts, :continuous => :eye_angle) # rename the column "continuous" to "eye_angle"
evts.animal[evts.animal.=="car"] .= "dog"
evts.animal[evts.animal.=="face"] .= "cat"

evts.continuous_random .= rand(size(evts, 1)) # add a new column "continuous_random" with random values
evts.vegetable .=
    ["tomato", "carrot"][1 .+ (evts.eye_angle.+10 .* rand(size(evts, 1)).>7.5)] # make random samples with a correlation of e.g. 0.5 to evts.continuous

b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5)
b2b_solver =
    (x, y) -> UnfoldDecode.solver_b2b(
        x,
        y;
        cross_val_reps = 5,
        solver_fun = UnfoldDecode.model_lasso,
    )
b2b_solver =
    (x, y) -> UnfoldDecode.solver_b2b(
        x,
        y;
        cross_val_reps = 5,
        solver_fun = UnfoldDecode.model_lsq,
    )
b2b_solver =
    (x, y) -> UnfoldDecode.solver_b2b(
        x,
        y;
        cross_val_reps = 5,
        solver_fun = UnfoldDecode.model_svm,
    )
b2b_solver =
    (x, y) -> UnfoldDecode.solver_b2b(
        x,
        y;
        cross_val_reps = 5,
        solver_fun = UnfoldDecode.model_ada,
    )


dat_3d = permutedims(repeat(dat, 1, 1, 20), [3 1 2]); # repeat the dat 20 times and permute the dimensions
dat_3d .+= 0.1 * rand(size(dat_3d)...)

#---
function run_b2b(f)
    designDict = [Any => (f, range(0, 0.44, step = 1 / 100))]
    m = Unfold.fit(UnfoldModel, designDict, evts, dat_3d; solver = b2b_solver)
    results = coeftable(m)
    results.estimate = abs.(results.estimate)
    results = results[results.coefname.!="(Intercept)", :]
    results.formula .= string(f)
    return results
end

results_all = DataFrame();
results_all = vcat(
    run_b2b(@formula 0 ~ 1 + animal + eye_angle),
    ## The first one takes `animal` and `eye_angle`, which are two independent independent variables that can impact the result into account
    run_b2b(@formula 0 ~ 1 + animal + vegetable),
    ## The second one takes `animal` and `vegetable`, which are two correlated independent variables in which only one variable really affects the result into account
    run_b2b(@formula 0 ~ 1 + animal + vegetable + eye_angle),
    ## The third one takes `animal` and `vegetable` and `eye_angle`, which are all variables mentioned above into account
    run_b2b(@formula 0 ~ 1 + animal + eye_angle + continuous_random + vegetable),
);

plot_erp(
    results_all;
    mapping = (; row = :formula),
    axis = (xlabel = "Time [s]", ylabel = "Performance"),
)
#---

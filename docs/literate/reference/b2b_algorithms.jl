using Unfold
using UnfoldMakie, CairoMakie
using UnfoldSim
using UnfoldDecode
using DataFrames
using Statistics
include("../../example_rename_events.jl")
# # Comparison of different solvers for G and H
# Let's prepare some data again
dat, evts = UnfoldSim.predef_eeg(; noiselevel = 0.1, return_epoched = true);
dat_3d = permutedims(repeat(dat, 1, 1, 20), [3 1 2]);
dat_3d .+= range(5, 20, size(dat_3d, 1)) .* rand(size(dat_3d)...); # vary the noise per channel
evts.correlated .=
    ["tomato", "carrot"][1 .+ (evts.continuous.+10 .* rand(size(evts, 1)).>7.5)];

# #### Comparison of the results from different regression methods
function run_b2b(solver_G, solver_H; kwargs...)
    b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; solver_G, solver_H, kwargs...)

    f = @formula(0 ~ 1 + condition + continuous + correlated)
    ## Define a design dictionary according to the formula
    times = range(0, 0.44, step = 1 / 100)
    designDict = [Any => (f, times)]
    ## Fit the model
    m = Unfold.fit(UnfoldModel, designDict, evts, dat_3d; solver = b2b_solver)

    results = coeftable(m)
    results.estimate = abs.(results.estimate)
    results = results[results.coefname.!="(Intercept)", :]
    results.solver_G .= string(solver_G)
    results.solver_H .= string(solver_H)
    return results
end;

run_models(list) = vcat(map(m -> run_b2b(m...; multithreading = true), list)...)
results_all = run_models([
    [UnfoldDecode.model_lsq, UnfoldDecode.model_lsq],
    [UnfoldDecode.model_xgboost, UnfoldDecode.model_lsq],
    [UnfoldDecode.model_ridge, UnfoldDecode.model_ridge],
    [UnfoldDecode.model_ridge, UnfoldDecode.model_lsq],
])

plot_erp(
    results_all;
    mapping = (; col = :solver_G, row = :solver_H),
    xis = (xlabel = "Time [s]", ylabel = "Performance"),
)



# Not all combinations are calculated due to time reason.

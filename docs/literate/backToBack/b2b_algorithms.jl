using Unfold
using UnfoldMakie, CairoMakie
using UnfoldSim
using UnfoldDecode
using DataFrames
using Statistics
include("../../example_rename_events.jl")
# # Comparison of different solvers for G and H
# Let's prepare some data again
dat, evts = UnfoldSim.predef_eeg(; noiselevel=0.1, return_epoched=true);
dat_3d = permutedims(repeat(dat, 1, 1, 20), [3 1 2]);
dat_3d .+= 0.1 * rand(size(dat_3d)...);
evts.correlated .= ["tomato", "carrot"][1 .+ (evts.continuous.+10 .* rand(size(evts, 1)).>7.5)];

# #### Comparison of the results from different regression methods
function run_b2b(solver_G, solver_H; kwargs...)
    b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; solver_G, solver_H, kwargs...)

    f = @formula(0 ~ 1 + condition + continuous + correlated)
    ## Define a design dictionary according to the formula
    times = range(0, 0.44, step=1 / 100)
    designDict = [Any => (f, times)]
    ## Fit the model
    m = Unfold.fit(UnfoldModel, designDict, evts, dat_3d; solver=b2b_solver)

    results = coeftable(m)
    results.estimate = abs.(results.estimate)
    results = results[results.coefname.!="(Intercept)", :]
    results.solver_G .= typeof(solver_G).name.name
    results.solver_H .= typeof(solver_H).name.name
    return results
end;

run_models(list) = vcat(map(m -> run_b2b(m...), list)...)
results_all = run_models([
    [UnfoldDecode.model_lsq, UnfoldDecode.model_lsq],
    #    [UnfoldDecode.model_svm, UnfoldDecode.model_svm]
    #[UnfoldDecode.model_lasso, UnfoldDecode.model_lasso],
    [UnfoldDecode.model_ridge, UnfoldDecode.model_ridge],])

plot_erp(results_all; mapping=(; col=:solver_G), xis=(xlabel="Time [s]", ylabel="Performance"))




# They fall into 3 categories, {Ridge, Lasso, LS}, {SVM}, and {Adaboost}.

# For the Ridge, Lasso, and LS regression, the results are almost the same and can clearly split the effect of different independent variables. However, these three methods share the same feature that there are peaks at certain time points, which we have no idea about it.

# For the SVM regression, the shape of the plot has two separate flattened peaks, which is the most ideal result.

# For the Adaboost regression, the result is similar to the SVM, but one problem with the result is that the value of 'continuous_random' is not 0 at the begining, which is not ideal. One possible reason is that the Adaboost algrithm choose random initial weight.


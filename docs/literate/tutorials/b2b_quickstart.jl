# # Setup
using Unfold
using UnfoldMakie, CairoMakie
using UnfoldSim
using UnfoldDecode
using DataFrames
using Statistics

# # Quick start of b2b solver

# ## Data generation
# #### Simulation and data collection
# Generate single channel data via UnfoldSim.jl
dat, evts = UnfoldSim.predef_eeg(; noiselevel = 0.1, return_epoched = true);
# `dat` is a time x repetition Matrix, `evts` is a `DataFrame``with independent variables / features to explain the data


# #### Dimension expansion
# Repeat the dat 20 times, representing 20 channels. In the future we will replace this with a direct multi-channel simulation.
dat_3d = permutedims(repeat(dat, 1, 1, 20), [3 1 2]);
# Channels shouldnt be identical, so we add some noise.
dat_3d .+= 0.1 * rand(size(dat_3d)...);

# ## Modeling

# #### Solver selection
# Call b2b solver in UnfoldDecode
b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5);
# !!! hint
#     one could specify the specific solvers for G and H by passing the `solver_G` and `solver_H` parameters to the `solver_b2b` function. Implemented solvers are ridge, lasso, lsq, svm, but other solvers from MLJ.jl can be used as well.

# #### Generate the formula
# We want to decode `condition`, but simultaneously control for the effect of `continuous`.
f = @formula 0 ~ 1 + condition + continuous
time = range(0, 0.44, step = 1 / 100)
designDict = [Any => (f, time)]

m = Unfold.fit(UnfoldModel, designDict, evts, dat_3d; solver = b2b_solver);

results = coeftable(m);
results.estimate = abs.(results.estimate); ## back2back has no sign
results = results[results.coefname.!="(Intercept)", :] ## the intercept in b2b is hard to interpret


#
# ## Plotting
plot_erp(results; axis = (xlabel = "Time [s]", ylabel = "Performance"))
#
# We can see from the graph that b2b solver identifies regions where the signal can be decoded, taking into account the `continuous` feature

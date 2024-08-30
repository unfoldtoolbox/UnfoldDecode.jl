using Unfold
using UnfoldMakie, CairoMakie
using UnfoldSim
using UnfoldDecode
using DataFrames
using Statistics

# # Quick start of b2b solver

# ## Data generation
# #### Simulation and data collection
# Collect the data genarated by UnfoldSim, and add certian level of noise
dat, evts = UnfoldSim.predef_eeg(; noiselevel = 0.1, return_epoched = true);
# `dat` is a 45 * 2000 matrix consisting of the result of the dependent variable of the simulation

# `evts` is a matrix consisting of the data of the independent variables during the simulaiton


# #### Dimension expansion
# Repeat the dat 20 times, representing 20 channels and therefore we have a new dimension and permute the dimensions for convenience
dat_3d = permutedims(repeat(dat, 1, 1, 20), [3 1 2]); 
# Add some noise to each channel to better simulate the result in real world
dat_3d .+= 0.1*rand(size(dat_3d)...);
# Now we have the final version of the dependent variable data needed for our model



# ## Modeling

# #### Solver selection
# Call the solver in UnfoldDecode

# Here we've accomplished link to 5 different methods for regression needed in our algorithm: Lasso, Ridge, LS, SVM, and Adaboost.

# They can be chosen by refering to the parameter `solver_fun`
b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5);
## b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5, solver_fun = UnfoldDecode.model_lasso);
## b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5, solver_fun = UnfoldDecode.model_lsq);
## b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5, solver_fun = UnfoldDecode.model_svm);
## b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5, solver_fun = UnfoldDecode.model_ada);
#


# #### Generate the formula

# It takes `condition` and `continuous`, which are two independent variables that can impact the result into account
f = @formula 0 ~ 1 + condition + continuous;
designDict = [Any => (f, range(0, 0.44, step = 1/100))];
m = Unfold.fit(UnfoldModel, designDict, evts, dat_3d; solver = b2b_solver);
results = coeftable(m);
results.estimate = abs.(results.estimate);
results = results[results.coefname .!="(Intercept)",:];

    
#
# ## Plotting
plot_erp(results; axis = (xlabel = "Time [s]", ylabel = "Performance"))
#
# We can see from the graph that b2b solver can separate the effect of different independent variables
# 
# As we can see from the graph, independent variable 'condition' has a significant effect between 0.1s and 0.2s, while 'continuous' has a significant effect between 0.2s and 0.4s


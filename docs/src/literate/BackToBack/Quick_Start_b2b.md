```@meta
EditURL = "Quick_Start_b2b.jl"
```

````@example Quick_Start_b2b
using Unfold
using UnfoldMakie, CairoMakie
using UnfoldSim
using UnfoldDecode
using DataFrames
using Statistics
````

# Quick start of b2b solver

## Data generation
#### Simulation and data collection
Collect the data genarated by UnfoldSim, and add certian level of noise

````@example Quick_Start_b2b
dat, evts = UnfoldSim.predef_eeg(; noiselevel = 0.1, return_epoched = true);
nothing #hide
````

`dat` is a 45 * 2000 matrix consisting of the result of the dependent variable of the simulation

`evts` is a matrix consisting of the data of the independent variables during the simulaiton

#### Dimension expansion
Repeat the dat 20 times, representing 20 channels and therefore we have a new dimension and permute the dimensions for convenience

````@example Quick_Start_b2b
dat_3d = permutedims(repeat(dat, 1, 1, 20), [3 1 2]);
nothing #hide
````

Add some noise to each channel to better simulate the result in real world

````@example Quick_Start_b2b
dat_3d .+= 0.1*rand(size(dat_3d)...);
nothing #hide
````

Now we have the final version of the dependent variable data needed for our model

## Modeling

#### Solver selection
Call the solver in UnfoldDecode

Here we've accomplished link to 3 different methods for regression needed in our algorithm: Ridge, LS, and SVM

They can be chosen by refering to the parameter `solver_fun`

````@example Quick_Start_b2b
b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5, solver_fun=UnfoldDecode.model_lsq);
# b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5, solver_fun="LSRegressor")
# b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5, solver_fun="SVMLinearRegressor")
````

#### Generate the formula

It takes `condition` and `continuous`, which are two independent independent variables that can impact the result into account

````@example Quick_Start_b2b
f = @formula 0 ~ 1 + condition + continuous;
designDict = [Any => (f, range(0, 0.44, step = 1/100))];
m = Unfold.fit(UnfoldModel, designDict, evts, dat_3d; solver = b2b_solver);
results = coeftable(m);
results.estimate = abs.(results.estimate);
results = results[results.coefname .!="(Intercept)",:];
nothing #hide
````

## Plotting

````@example Quick_Start_b2b
plot_erp(results; axis = (xlabel = "Time [s]", ylabel = "Performance"))
````

We can see from the graph that b2b solver can separate the effect of different independent variables

As we can see from the graph, independent variable 'condition' has a significant effect between 0.1s and 0.2s, while 'continuous' has a significant effect between 0.2s and 0.4s

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*


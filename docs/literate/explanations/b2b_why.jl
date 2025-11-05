using Unfold
using UnfoldMakie, CairoMakie
using UnfoldSim
using UnfoldDecode
using DataFrames
using Statistics
include("../../../example_rename_events.jl")

#
# # [Motivation for BacktoBack](@id explainer-b2b)
# ## Introduction

# “Back-to-Back” regression (B2B) is an approach to estimate the decoding performance from a set of correlated factors.
# Why do we need this? Let's have a look at a simple example:
# ![My Image](./assets/dog_and_cat.png)
#
# Imagine we record EEG data from an eyetracking experiment, and investigate each fixation (a resting period of the eye) as an event for an ERP.
# Imagine, we have both cats and dogs, but that we also make large and small eye-movements.
# ## Data generation

# #### Simulation and data collection
# Collect the data genarated by UnfoldSim, and add certian level of noise
dat, evts = UnfoldSim.predef_eeg(; noiselevel = 0.1, return_epoched = true);
evts = example_rename_events(evts)

# #### Data further generating
# To make the example more impressive, let's add an orthogonal variable `vegetable`. But this variable is special:
# It is correlated with the covariate `eye_movement_size`.
evts.vegetable .=
    ["tomato", "carrot"][1 .+ (evts.eye_movement_size .+ 10 .* rand(size(evts, 1)) .> 7.5)];
cor(evts.eye_movement_size, evts.vegetable .== "carrot")

# ![My Image](./assets/dog_and_cat_and_vegetable.png)

# Summarized, we have three independent variables: `animal`, `eye_movement_size`, and `vegetable`, with the latter two being correlated.

# !!! important
#     By construction, there is no `vegetable` effect in the data! All effects we find for vegetable solely come from the correlation with `eye_movement_size` - Now imagine decoding vegetable, you would find a strong **spurious** effect.


# #### Making it multi channel
# Decoding is multivariate, so we need multiple channels. For simplicity, we just repeat the data 20 times, representing 20 channels.
dat_3d = permutedims(repeat(dat, 1, 1, 20), [3 1 2]);
dat_3d .+= 0.1 * rand(size(dat_3d)...);

# #### Solver selection
b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5);

# #### Fitting function
# Because we want to compare different scenarios involving different variables, we define a function to estimate them.
function run_b2b(f)
    ## Define a design dictionary according to the formula
    times = range(0, 0.44, step = 1 / 100)
    designDict = [Any => (f, times)]
    ## Fit the model
    m = Unfold.fit(UnfoldModel, designDict, evts, dat_3d; solver = b2b_solver)

    results = coeftable(m)
    results.estimate = abs.(results.estimate)
    results = results[results.coefname .!= "(Intercept)", :]
    results.formula .= string(f)
    return results
end;

# let's run a decoder without accounting for the other factors

results_all = map(
    run_b2b,
    [
        @formula(0 ~ 1 + animal),
        @formula(0 ~ 1 + vegetable),
        @formula(0 ~ 1 + eye_movement_size)
    ],
)
plot_erp(vcat(results_all...); axis = (xlabel = "Time [s]", ylabel = "Performance"))

# As one can see, all three variables can be decoded well. **Even though vegetable had no effect on the data!!**

# Let's now use B2B to take into account the correlation between `vegetable` and `eye_movement_size`:

results_all = map(
    run_b2b,
    [@formula(0 ~ 1 + vegetable), @formula(0 ~ 1 + vegetable + eye_movement_size)],
)
plot_erp(
    vcat(results_all...);
    mapping = (; color = :coefname, row = :formula),
    axis = (xlabel = "Time [s]", ylabel = "Performance"),
)

# As can be seen, when modelling both effects (lower plot), the vegetable effect (correctly and as intended) vanishes. We now learned, that decodable information is only in  `eye_movement_size`, but not `vegetable`, which is "just" correlated

# For completeness sake, we also include the comparison to a non-correlated effect
results_all = map(
    run_b2b,
    [@formula(0 ~ 1 + animal), @formula(0 ~ 1 + animal + vegetable + eye_movement_size)],
)
plot_erp(
    vcat(results_all...);
    mapping = (; color = :coefname, row = :formula),
    axis = (xlabel = "Time [s]", ylabel = "Performance"),
)

# the animal effect remains untouched :)

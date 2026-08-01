using UnfoldDecode
using UnfoldSim
using UnfoldMakie
using CairoMakie
using Unfold

# # Overlap-corrected decoding
# We will try to introduce as many fancy features as possible
# Please read the "tutorial" first

# ## Simulation
# multi-event
dat, evt = UnfoldSim.predef_eeg()
evt.event = rand(["eventA", "eventB"], size(evt, 1)) # add random events
dat = repeat(dat', 5)
dat .= dat .+ 20 .* rand(size(dat)...)

# ## Overlap-model Definition
# We have two basis functions now, with two different timewindows. Let's see if it works!
des = [
    "eventA" => (@formula(0 ~ 1 + condition + continuous), firbasis((-0.1, 1.0), 100)),
    "eventB" => (@formula(0 ~ 1 + continuous), firbasis((-0.3, 0.5), 100)),
]
# To show that it is possible, we explicitly specify the solver
customsolver = (x, y) -> Unfold.solver_default(x, y)
uf = Unfold.fit(UnfoldModel, des, evt, dat[1, :]; solver = customsolver);
plot_erp(coeftable(uf); mapping = (; col = :eventname))

# ## Fitting the Overlap-corrected LDA model
using MLJ, MultivariateStats, MLJMultivariateStatsInterface
LDA = @load LDA pkg = MultivariateStats

# you could use other parameters, check out `?LDA`
ldaModel = LDA(
    method = :whiten,
    cov_w = SimpleCovariance(),
    cov_b = SimpleCovariance(),
    regcoef = 1e-3,
)

uf_lda = UnfoldDecode.fit(
    UnfoldDecodingModel,
    des,
    evt,
    dat,
    ldaModel,
    "eventA" => :condition;
    nfolds = 2,# only 2 folds to speed up computation
    unfold_fit_options = (; solver = customsolver), #customer solver for fun
    eventcolumn = :event, # actually the default, but maybe your event dataframe has a different name?
    multithreading = false,
) # who needs speed anyway :shrug:

plot_erp(coeftable(uf_lda))

# Voila, the model classified the correct period at the correct event

# ## Fitting the Overlap-corrected Ridge Regression model
RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels

# you could use other parameters, check out `?RidgeRegressor`
ridgeModel = RidgeRegressor(lambda = 1.0, fit_intercept = true, penalize_intercept = false)

# change the setting in MLJ to allow for continuous predictions

MLJ.machine(
    model::RidgeRegressor,
    X::AbstractMatrix{Float64},
    y::SubArray{Float64};
    kwargs...,
) = MLJ.machine(model, MLJ.table(X), y; kwargs...)

# load the model
uf_ridge = Unfold.fit(
    UnfoldDecodingModel,
    des,
    evt,
    dat,
    ridgeModel,
    "eventA" => :continuous;
    nfolds = 2,
    predict_type = Continuous,
)


ridge_scores = coeftable(uf_ridge, measure = RSquared())
plot_erp(ridge_scores; mapping = (; color = :estimate))

# Voila again, the model can predict the correct period at the correct event

# ## Grid search for ridge regression lambda parameter

# We can use MLJ's built-in tuning functionality to perform a grid search over the lambda parameter for ridge regression.
r = range(ridgeModel, :lambda; lower = 1e-6, upper = 1e2, scale = :log)

ridgeTunedModel = TunedModel(
    model = ridgeModel,
    resampling = CV(nfolds = 3),
    range = r,
    tuning = Grid(resolution = 4), # test 4 λ values on a logarithmic scale
    measure = RSquared(),
)


# Now we can fit the model with the tuned hyperparameter
uf_ridge_tuned = Unfold.fit(
    UnfoldDecodingModel,
    des,
    evt,
    dat,
    ridgeTunedModel,
    "eventA" => :continuous;
    nfolds = 2,
    predict_type = Continuous,
)

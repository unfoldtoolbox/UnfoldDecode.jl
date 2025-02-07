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

# test cxcxcxcxcxcxcccccccccccccccccccccccccccccccccccccc

# ## Overlap-model Definition
# We have two basis functions now, with two different timewindows. Let's see if it works!
des = ["eventA" => (@formula(0 ~ 1 + condition + continuous), firbasis((-0.1, 1.0), 100)),
    "eventB" => (@formula(0 ~ 1 + continuous), firbasis((-0.3, 0.5), 100))]
# To show that it is possible, we explicitly specify the solver
customsolver = (x, y) -> Unfold.solver_default(x, y)
uf = Unfold.fit(UnfoldModel, des, evt, dat[1, :]; solver=customsolver);
plot_erp(coeftable(uf); mapping=(; col=:eventname))

# ## Fitting the Overlap-corrected LDA model
using MLJ, MultivariateStats, MLJMultivariateStatsInterface
LDA = @load LDA pkg = MultivariateStats

# you could use other parameters, check out `?LDA`
ldaModel = LDA(method=:whiten, cov_w=SimpleCovariance(), cov_b=SimpleCovariance(), regcoef=1e-3)

uf_lda = UnfoldDecode.fit(UnfoldDecodingModel, des, evt, dat, ldaModel, "eventA" => :condition;
    nfolds=2,# only 2 folds to speed up computation
    unfold_fit_options=(; solver=customsolver), #customer solver for fun
    eventcolumn=:event, # actually the default, but maybe your event dataframe has a different name?
    multithreading=false) # who needs speed anyway :shrug:

plot_erp(coeftable(uf_lda))

# Voila, the model classified the correct period at the correct event


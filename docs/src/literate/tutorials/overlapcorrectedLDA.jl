using UnfoldDecode
using UnfoldSim
using UnfoldMakie
using CairoMakie

# # Overlap-corrected decoding
# This approach follows the work of the Deoulle Group, especially Gal Vishne's work as published 2023: https://doi.org/10.1101/2023.06.28.546397

# ## Simulation
# We start with simulating some continuous, overlapping data with two conditions. 
# As of now, `UnfoldSim` doesnt support multichannel, so we simply repeat the channel and add some noise`
dat,evt = UnfoldSim.predef_eeg(MersenneTwister(1))
dat = repeat(dat',20)
dat .= dat .+ 20 .* rand(MersenneTwister(1),size(dat)...)

# ## Overlap-model Definition
# We have to define what model we want to use for overlap correction
# We decide for a one-basisfunction model, with one covariate, from `-0.1` to `0.5` seconds afte the stimulus onset. Sampling rate `100` as in the simulation
des = Dict(Any => (@formula(0~1+condition+continuous),firbasis((-0.1,0.5),100)));
# Fitting and visualizing the model
uf = Unfold.fit(UnfoldModel,des,evt,dat[1,:]);
plot_erp(coeftable(uf))

# ## Fitting the Overlap-corrected LDA model
uf_lda = fit(UnfoldDecodingModel,des,evt,dat,LDA(),Any=>:condition)
plot_erp(coeftable(uf_lda))

# Voila, the model classified the correct period. 




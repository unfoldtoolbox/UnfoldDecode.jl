using UnfoldSim 
using UnfoldMakie
using CairoMakie
using DataFrames
using Random

design = SingleSubjectDesign(conditions = Dict(:picture => ["dog","cat"],:hue =>["grayscale","color"])) |> x->RepeatDesign(x,10)

c = LinearModelComponent(; basis = p100(), formula = @formula(0 ~ 1+picture), β = [1,0.5]);
c2 = LinearModelComponent(; basis = p300(), formula = @formula(0 ~ 1+picture), β = [1,-3]);

hart = headmodel(type = "hartmut")
mc = UnfoldSim.MultichannelComponent(c, hart => "Left Postcentral Gyrus")

mc2 = UnfoldSim.MultichannelComponent(c2, hart => "Right Occipital Pole")
mc2 = UnfoldSim.MultichannelComponent(c2, hart => "Left Postcentral Gyrus")

onset = NoOnset();#UniformOnset(; width = 20, offset = 4);

data, events =
    simulate(MersenneTwister(1), design, [mc, mc2], onset, PinkNoise(noiselevel = 0.5);return_epoched=true
    )
size(data)

pos3d = hart.electrodes["pos"];
pos2d = to_positions(pos3d')
pos2d = [Point2f(p[1] + 0.5, p[2] + 0.5) for p in pos2d];

#---
using Statistics
f = Figure()
data_m = mean(data,dims=3)
df = DataFrame(
    :estimate => data_m[:],
    :channel => repeat(1:size(data_m, 1), outer = size(data_m, 2)),
    :time => repeat(1:size(data_m, 2), inner = size(data_m, 1)),
)
plot_butterfly!(f[1, 1:2], df; positions = pos2d)
plot_topoplot!(
    f[2, 1],
    df[df.time.==12, :];
    positions = pos2d,
    visual = (; enlarge = 0.5, label_scatter = true),
    axis = (; limits = ((0, 1), (0, 0.9))),
)
plot_topoplot!(
    f[2, 2],
    df[df.time.==30, :];
    positions = pos2d,
    visual = (; enlarge = 0.5, label_scatter = false),
    axis = (; limits = ((0, 1), (0, 0.9))),
)
f

#---

data_d = mean(data[:,:,events.picture .=="dog"],dims=3) .- mean(data[:,:,events.picture .=="cat"],dims=3)
#data_d = mean(data[:,:,events.hue .=="color"],dims=3) .- mean(data[:,:,events.hue .=="grayscale"],dims=3)

f = Figure()
df = DataFrame(
    :estimate => data_d[:],
    :channel => repeat(1:size(data_d, 1), outer = size(data_d, 2)),
    :time => repeat(1:size(data_d, 2), inner = size(data_d, 1)),
)
plot_butterfly!(f[1, 1:2], df; positions = pos2d)


f
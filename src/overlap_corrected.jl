"""
Returns the single trial corrected (and "epoched") trials from the basis `target_eventname`

"""
function singletrials(
    dat,
    uf_train::UnfoldLinearModelContinuousTime,
    tbltrain::AbstractDataFrame,
    target_eventname,
    eventcolumn,
)



    basisnames = Unfold.basisname(uf_train)
    ix = findfirst(target_eventname .== basisnames)

    tw = Unfold.calc_epoch_timewindow(uf_train, target_eventname)


    find_lat =
        x -> x[
            target_eventname == Any ? (1:size(x, 1)) :
            x[:, eventcolumn] .== target_eventname,
            :latency,
        ]
    latencies = vcat(find_lat(tbltrain))


    pred_target = Unfold.predict_no_overlap(
        uf_train,
        coef(uf_train),
        Unfold.formulas(uf_train)[ix:ix],
        [tbltrain[tbltrain[:, eventcolumn].==target_eventname, :]],
    )[1]

    pred_full = Unfold.predict_partial_overlap(
        uf_train,
        coef(uf_train),
        [tbltrain];
        epoch_to = target_eventname,
        eventcolumn,
    )
    residuals = Unfold._residuals(pred_full, dat, latencies, (tw[1], tw[end]))

    @debug size(residuals) size(pred_target)
    #return pred, dat, latencies, (tw[1], tw[end])
    return residuals .+ pred_target
end

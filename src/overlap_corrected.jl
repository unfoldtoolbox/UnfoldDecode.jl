"""
Returns the single trial corrected (and "epoched") trials from the basis `target_eventname`

"""
function singletrials(dat, uf_train::UnfoldLinearModelContinuousTime,
    tbltrain::AbstractDataFrame, target_eventname, eventcolumn)


    #pred_notarget = Unfold.predict_partial_overlap(uf_train, coef(uf_train), [tbltrain]; exclude_basis=[target_eventname], epoch_to=target_eventname, eventcolumn)
    basisnames = Unfold.basisname(uf_train)
    ix = findfirst(target_eventname .== basisnames)
    pred = Unfold.predict(uf_train, overlap=false)[ix]
    tw = Unfold.calc_epoch_timewindow(uf_train, target_eventname)


    find_lat =
        x -> x[
            target_eventname == Any ? (1:size(x, 1)) : x[:, eventcolumn] .== target_eventname,
            :latency,
        ]
    latencies = vcat(find_lat(tbltrain))

    resid_targetonly = Unfold._residuals(pred, dat, latencies, (tw[1], tw[end]))

    return pred, dat, latencies, (tw[1], tw[end])
    return resid_targetonly
end

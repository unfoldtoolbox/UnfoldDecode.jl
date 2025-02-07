
"""
using Unfold: @maybe_threads
Use UnfoldLinearModelContinuous time to remove all overlap from an event. Use the `model` classifier to predict the EEG data.
Everything runs cross-validated with `nfolds`

- `model`: By default LDA(), but could be any MLJ machine with specified parameters
- `target`: String or Symbol with the `tbl[:,column]` to be decoded
- `unfold_fit_options`: optional Named Tuple as kwargs to provide to the initial "overlap-cleaning" modelfit, e.g. `unfold_fit_options = (;solver=(x,y)->solver_krylov(x,y,GPU=true))` for GPU fit (need to load `Krylov` and `CUDA` before)
"""
function Unfold.fit(
    UnfoldDecodingModel,
    design,
    tbl,#::DataFrame,
    dat::AbstractMatrix,
    model::MLJ.Model,
    target::Pair;
    nfolds = 6,
    eventcolumn = :event,
    unfold_fit_options = (;),
    multithreading = true,
)

    tbl = deepcopy(tbl)
    # sort to split by neighbouring samples for overlap
    sort!(tbl, :latency)
    if first(target) == Any
        tbl.event .= Any
    end
    # get CV splits
    train_test = MLJBase.train_test_pairs(CV(; nfolds = nfolds), 1:size(tbl, 1))


    fits = Array{DecodingFit}(undef, length(train_test))
    #Unfold.@maybe_threads multithreading
    for split = 1:length(train_test)


        tbltrain = @view tbl[train_test[split][1], :]
        tbltest = @view tbl[train_test[split][2], :]

        # XXX remove the boundary data to ensure no leakage
        uf_train = Unfold.fit(
            UnfoldLinearModelContinuousTime,
            design,
            tbl,
            dat;
            eventcolumn = eventcolumn,
            unfold_fit_options...,
        )

        uf_test = Unfold.fit(
            UnfoldLinearModelContinuousTime,
            design,
            tbltest,
            dat;
            eventcolumn = eventcolumn,
            unfold_fit_options...,
        )

        # get overlap free single trails
        X_train = singletrials(dat, uf_train, tbltrain, target[1], eventcolumn)
        X_test = singletrials(dat, uf_test, tbltest, target[1], eventcolumn)

        # get overlap free single trails test
        ix_train =
            first(target) == Any ? (1:size(tbltrain, 1)) :
            tbltrain[:, eventcolumn] .== target[1]
        ix_test =
            target[1] == Any ? (1:size(tbltest, 1)) : tbltest[:, eventcolumn] .== target[1]

        y_train = coerce(tbltrain[ix_train, target[2]], OrderedFactor)
        y_test = coerce(tbltest[ix_test, target[2]], OrderedFactor)


        # remove missing

        # train

        missingIx = .!any(ismissing.(X_train), dims = (1, 2))
        goodIx = dropdims(missingIx, dims = (1, 2))

        machines = fit_timepoints(
            model,
            disallowmissing(@view(X_train[:, :, goodIx])),
            @view(y_train[goodIx]),
        )

        # test
        missingIx = .!any(ismissing.(X_test), dims = (1, 2))
        goodIx = dropdims(missingIx, dims = (1, 2))
        yhat = predict_timepoints(machines, disallowmissing(@view(X_test[:, :, goodIx])))

        # save it
        times = Unfold.times(uf_train)[1]
        #return machines, train_test[split], yhat, y_test, times
        fits[split] = DecodingFit(machines, train_test[split], yhat, y_test[goodIx], times)



    end
    #return fits
    return UnfoldDecodingModel(design, target, tbl, fits)
end


"""
Use UnfoldLinearModelContinuous time to remove all overlap from an event. Use the `model` classifier to predict the EEG data.
Everything runs cross-validated with `nfolds`

- `model`: By default LDA(), but could be any MLJ machine with specified parameters
- `target`: String or Symbol with the `tbl[:,column]` to be decoded
- `UnfoldFitkwargs`: optional Named Tuple as kwargs to provide to the initial "overlap-cleaning" modelfit, e.g. `UnfoldFitkwargs = (;solver=(x,y)->solver_krylov(x,y,GPU=true))` for GPU fit (need to load `Krylov` and `CUDA` before)
"""
function fit(UnfoldDecodingModel,design::Dict,
tbl,#::DataFrame,
dat::AbstractMatrix,
model::MLJ.Model,
target::Pair;
nfolds=6,eventcolumn=:event,UnfoldFitkwargs=(;))

# sort to split by neighbouring samples for overlap
sort!(tbl,:latency)

# get CV splits
train_test = MLJBase.train_test_pairs(CV(;nfolds=nfolds),1:size(tbl,1))


fits = Array{DecodingFit}(undef,length(train_test))
for split = 1:length(train_test)
    

    tbltrain = tbl[train_test[split][1],:]
    tbltest =  tbl[train_test[split][2],:]
    
    # XXX remove the boundary data to ensure no leakage
    uf_train = Unfold.fit(UnfoldLinearModelContinuousTime,
        design,tbl,dat;eventcolumn=eventcolumn,UnfoldFitkwargs...)
    uf_test  = Unfold.fit(UnfoldLinearModelContinuousTime,
        design,tbltest,dat;eventcolumn=eventcolumn,UnfoldFitkwargs...)
    
    # get overlap free single trails 
    X_train = singletrials(uf_train,tbltrain,target[1],eventcolumn)
    X_test  = singletrials(uf_test, tbltest, target[1],eventcolumn)

    # get overlap free single trails test
    y_train = coerce(tbltrain[:,target[2]],OrderedFactor)
    y_test  = coerce(tbltest[:, target[2]],OrderedFactor)

    # train
     machines =fit_timepoints(model,X_train,y_train)

    # test
    yhat = predict_timepoints(machines,X_test)

    # save it
    times = Unfold.times(Unfold.design(uf_train)[target[1]][2])[1:end-1]
    fits[split] = DecodingFit(machines,train_test[split],yhat,y_test,times)

        

end
    
return UnfoldDecodingModel(design, target,tbl,fits)
end
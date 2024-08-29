"""
For each time-point, fit a MLJ-Machine
"""
function fit_timepoints(model,X_train,y_train)
    logger=Logging.SimpleLogger(stderr,Logging.Warn)
    X_train = coerce(X_train,Continuous)
    machines = Array{MLJ.Machine}(undef,size(X_train,2))
    for t = 1:size(X_train,2)
        machines[t] = machine(
                model,
                @view(X_train[:,t,:])',
                y_train;
                scitype_check_level=0)
    
        Logging.with_logger(logger) do
            # fit(machine)
            MLBase.fit!(machines[t], verbosity=0) # Use MLBase.fit! to specify the fit function we used
        end
    end
    return machines
end	




function predict_timepoints(machines::Array{MLJ.Machine},X::AbstractArray)
    yhat = Array{AbstractVector}(undef,length(machines))
    for i = 1:length(machines)
        mach = machines[i]
        Xi = @view X[:,i,:]
        yhat[i] = MLJ.predict(mach,collect(Xi)')
    end
    return yhat
end
using Unfold
using UnfoldMakie, CairoMakie
using UnfoldSim
using UnfoldDecode
using DataFrames
using ProgressMeter
using LinearAlgebra
using Logging # to deactivate some MLJ output
using MLJ
using MLBase
using MLJLinearModels,Tables
using MLJScikitLearnInterface
using MLJMultivariateStatsInterface
using MultivariateStats

# ## solver_b2b function: 
## Removing time periods or samples that contain missing values
function solver_b2b(X,data::AbstractArray{T,3};kwargs...) where {T<:Union{Missing,<:Number}}
    X, data = drop_missing_epochs(X, data) 
    solver_b2b(X,data; kwargs...)
end;

## We set the default times of cross validation repetition is 10, and default regularization_method is "RidgeRegressor"
function solver_b2b(                     
        X, # design matrix
        data::AbstractArray{T,3};  
        cross_val_reps = 10,
        multithreading = true,
        show_progress = true,
        regularization_method::String="RidgeRegressor",
        kwargs...
    ) where {T<:Number}

## #### Choosing method
## if the chosen regularization_method is `RidgeRegressor`, we build a model based on RidgeRegressor and tune it
    if regularization_method == "RidgeRegressor"
        @load RidgeRegressor pkg=MLJLinearModels
        model = MLJLinearModels.RidgeRegressor(fit_intercept=false)
        tm = tunemodel(model;kwargs...)
## if the chosen regularization_method is `SVMRegressor`, we build a model based on SVMLinearRegressor (without tuning)
    elseif regularization_method == "SVMRegressor"
        @load SVMLinearRegressor pkg=MLJScikitLearnInterface
        tm = MLJScikitLearnInterface.SVMLinearRegressor()
    end;
## if the chosen regularization_method is `LSRegressor`, actually we don't need a model here for simplicity

    E = zeros(size(data,2),size(X,2),size(X,2))
    W = Array{Float64}(undef,size(data,2),size(X,2),size(data,1))
    prog = Progress(size(data, 2) * cross_val_reps;dt=0.1,enabled=show_progress)
    println("n = samples = $(size(X,1)) = $(size(data,3))")
    @showprogress 0.1 for t in 1:size(data,2)        
    
        for m in 1:cross_val_reps
            k_ix = collect(Kfold(size(data,3),2))
            Y1 = data[:,t,k_ix[1]]
            Y2 = data[:,t,k_ix[2]]
            X1 = X[k_ix[1],:]
            X2 = X[k_ix[2],:]
            if regularization_method == "LSRegressor"
                G = (Y1' \ X1)
                H = X2 \ (Y2' * G)
            elseif regularization_method == "RidgeRegressor" 
                G = solver_tune(tm,Y1',X1)
                H = solver_tune(tm,X2, (Y2'*G))
            elseif regularization_method == "SVMRegressor"
                println(regularization_method)
                G = solver_notune(tm,Y1',X1)
                H = solver_notune(tm,X2, (Y2'*G))
            end
            E[t,:,:] = E[t,:,:] + Diagonal(H[diagind(H)])
            ProgressMeter.next!(prog; showvalues = [(:time, t), (:cross_val_rep, m)])
        end
        E[t,:,:] = E[t,:,:] ./ cross_val_reps
        W[t,:,:] = (X*E[t,:,:])' / data[:,t,:]
    end
    ## extract diagonal
    beta = mapslices(diag,E,dims=[2,3])
    ## reshape to conform to ch x time x pred
    beta = permutedims(beta,[3 1 2])
    modelinfo = Dict("W"=>W,"E"=>E,"cross_val_reps"=>cross_val_reps) # no history implemented (yet?)
    return Unfold.LinearModelFit(beta, modelinfo)
end;

## We build the function for fetching the better hyperparameters of the model, which is needed in RidgeRegression
function tunemodel(model;nfolds=5,resolution = 10,measure=MLJ.rms,kwargs...)
    range = Base.range(model, :lambda, lower=1e-2, upper=1000, scale=:log10)
    tm = TunedModel(model=model,
                    resampling=CV(nfolds=nfolds),
                    tuning=Grid(resolution=resolution),
                    range=range,
                    measure=measure)
    return tm
end;

## We build the function for solvers in need of tuning
function solver_tune(tm,data,X)
    G = Array{Float64}(undef,size(data,2),size(X,2))
    for pred in 1:size(X,2)
        mtm = machine(tm,table(data),X[:,pred])
        fit!(mtm,verbosity=0)
        G[:,pred] = Tables.matrix(fitted_params(mtm).best_fitted_params.coefs)[:,2]
    end
    return G
end;

## We build the function for solvers no need of tuning
function solver_notune(tm,data,X)
    G = Array{Float64}(undef,size(data,2),size(X,2))
    for pred in 1:size(X,2)
        mtm = machine(tm,table(data),X[:,pred])
        fit!(mtm,verbosity=0)
        G[:,pred] = fitted_params(mtm).coef
    end
    return G
end;
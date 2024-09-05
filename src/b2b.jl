# Code currently duplicated in Unfold.jl
# https://github.com/unfoldtoolbox/Unfold.jl/edit/main/src/solver.jl
using Unfold: @maybe_threads

# Basic implementation of https://doi.org/10.1016/j.neuroimage.2020.117028
function solver_b2b(X, data::AbstractArray{T,3}; kwargs...) where {T<:Union{Missing,<:Number}}
    X, data = drop_missing_epochs(X, data)
    solver_b2b(X, data; kwargs...)
end

# predefined model functions
function model_lsq(; kwargs...)
    return (x, y) -> x \ y
end

function model_ridge(; kwargs...)
    model = gen_model_ridge()
    tm = tunemodel(model; kwargs...)
    return (x, y) -> _solve(tm, x, y)
end

function model_lasso(; kwargs...)
    model = gen_model_lasso()
    tm = tunemodel(model; kwargs...)
    return (x, y) -> _solve(tm, x, y)
end

function model_svm(; kwargs...)
    model = gen_model_svm()
    return (x, y) -> _solve(model, x, y)
end

"""
    solver_b2b(X, data::AbstractArray{T,3}; cross_val_reps=10, multithreading=true, show_progress=true, solver_G=model_ridge, solver_H=model_ridge, kwargs...)



"""
function solver_b2b(                     # when T is a number, 
    X, # design matrix
    data::AbstractArray{T,3};  # form a 3D array of data with type "T"
    cross_val_reps=10,
    multithreading=true,
    show_progress=true,
    solver_G=model_ridge,
    solver_H=model_ridge,
    kwargs...
) where {T<:Number}
    model_fun_G = solver_G(; kwargs...)
    model_fun_H = solver_H(; kwargs...)
    #@show model_fun

    E = zeros(size(data, 2), size(X, 2), size(X, 2))
    W = Array{Float64}(undef, size(data, 2), size(X, 2), size(data, 1))
    prog = Progress(size(data, 2) * cross_val_reps; dt=0.1, enabled=show_progress)
    @maybe_threads multithreading for t in 1:size(data, 2)

        for m in 1:cross_val_reps
            k_ix = collect(Kfold(size(data, 3), 2))
            Y1 = data[:, t, k_ix[1]]
            Y2 = data[:, t, k_ix[2]]
            X1 = X[k_ix[1], :]
            X2 = X[k_ix[2], :]

            G = model_fun_G(Y1', X1)
            H = model_fun_H(X2, Y2'G)

            E[t, :, :] = E[t, :, :] + Diagonal(H[diagind(H)])
            ProgressMeter.next!(prog; showvalues=[(:time, t), (:cross_val_rep, m)])
        end
        E[t, :, :] = E[t, :, :] ./ cross_val_reps
        W[t, :, :] = (X * E[t, :, :])' / data[:, t, :]
    end

    # extract diagonal
    beta = mapslices(diag, E, dims=[2, 3])
    # reshape to conform to ch x time x pred
    beta = permutedims(beta, [3 1 2])
    modelinfo = Dict("W" => W, "E" => E, "cross_val_reps" => cross_val_reps) # no history implemented (yet?)
    return Unfold.LinearModelFit(beta, modelinfo)
end

function gen_model_ridge()
    @load RidgeRegressor pkg = MLJLinearModels verbosity = 0
    model = MLJLinearModels.RidgeRegressor(fit_intercept=false)
    return model
end

function gen_model_lasso(; kwargs...)
    @load LassoRegressor pkg = MLJLinearModels verbosity = 0
    model = MLJLinearModels.LassoRegressor(fit_intercept=false)
    return model
end

function gen_model_svm(; kwargs...)
    @load SVMLinearRegressor pkg = MLJScikitLearnInterface
    model = MLJScikitLearnInterface.SVMLinearRegressor()
    return model
end


# tuning the hyperparameters of the model
function tunemodel(model; nfolds=5, resolution=10, measure=MLJ.rms, kwargs...)
    range = Base.range(model, :lambda, lower=1e-2, upper=1000, scale=:log10)
    tm = TunedModel(model=model,
        resampling=CV(nfolds=nfolds),
        tuning=Grid(resolution=resolution),
        range=range,
        measure=measure)
    return tm
end

# used to calculate the G and H
function _solve(tm, data, X)
    G = Array{Float64}(undef, size(data, 2), size(X, 2))
    for pred in 1:size(X, 2)
        mtm = machine(tm, table(data), X[:, pred])
        MLBase.fit!(mtm, verbosity=0)
        G[:, pred] = _fitted_param(mtm)


    end
    return G
end
_fitted_param(mtm::Machine) = Tables.matrix(fitted_params(mtm).best_fitted_params.coefs)[:, 2]
_fitted_param(mtm) = fitted_params(mtm).coef


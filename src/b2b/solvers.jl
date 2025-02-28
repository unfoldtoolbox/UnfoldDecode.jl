
# predefined model functions
model_lsq(X, y; kwargs...) = X \ y
model_lsq(X, ytrain, Xtest) = Xtest * (X \ ytrain)


model_ridge(X, ytrain, Xtest; kwargs...) = Xtest * model_ridge(X, ytrain; kwargs...)
function model_ridge(X, ytrain; kwargs...)
    @load RidgeRegressor pkg = MLJLinearModels verbosity = 0
    model = MLJLinearModels.RidgeRegressor(fit_intercept = false)
    tm = tunemodel(model; kwargs...)
    return _solve(tm, X, ytrain)
end

model_lasso(X, ytrain, Xtest; kwargs...) = Xtest * model_lasso(X, ytrain; kwargs...)
function model_lasso(X, ytrain; kwargs...)
    @load LassoRegressor pkg = MLJLinearModels verbosity = 0
    model = MLJLinearModels.LassoRegressor(fit_intercept = false)
    tm = tunemodel(model; kwargs...)
    return _solve(tm, X, ytrain)

end

function model_xgboost(X, ytrain, Xtest; kwargs...)
    #@debug "xgboost sizes" size(ytrain) size(X) size(Xtest)

    return map(
        (y) -> XGBoost.predict(
            xgboost(
                (X, y),
                num_round = 5,
                max_depth = 6,
                objective = "reg:squarederror",
                watchlist = [], # to silence the info output
                kwargs...,
            ),
            Xtest,
        ),
        eachcol(ytrain),
    ) |> x -> reduce(hcat, x)
end

function gen_model_svm(; kwargs...)
    #@load SVMLinearRegressor pkg = MLJScikitLearnInterface verbosity = 0
    #model = MLJScikitLearnInterface.SVMLinearRegressor()
    return model
end

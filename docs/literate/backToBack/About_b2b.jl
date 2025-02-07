using Unfold
using UnfoldDecode
using ProgressMeter
using LinearAlgebra
using MLJ
using MLBase
using MLJLinearModels, Tables
using MLJScikitLearnInterface
using MLJMultivariateStatsInterface
using MultivariateStats

# ## Main function

# First, we do the preliminary processing by removing time periods or samples that contain missing values
function solver_b2b(
    X,
    data::AbstractArray{T,3};
    kwargs...,
) where {T<:Union{Missing,<:Number}}
    X, data = drop_missing_epochs(X, data)
    solver_b2b(X, data; kwargs...)
end;

# Then, we start to define our main function.
# When T is a number:
function solver_b2b(
    ## Read in the design matrix `X`
    X,
    ## And a 3D array of data with type T `data`
    data::AbstractArray{T,3};
    ## We set the default times of cross validation repetition to be 10
    cross_val_reps = 10,
    multithreading = true,
    show_progress = true,
    ## And default method to be "RidgeRegressor"
    solver_fun = model_ridge,
    kwargs...,
) where {T<:Number}

    ## To make it clear for users, here we print out and retrive the name of the chosen solver function
    println("solver_fun = $solver_fun")
    model_fun = solver_fun(; kwargs...)

    ## Now we start defining the key part of our function according to the idea of BacktoBack algorithm

    ## First, we define a zero matrix E with suitable size.

    ## `size(data,2)`is the number of channels, that is, the number of dependent variables.

    ## `size(X,2)` is the number of factors, that is, the number of independent variables.

    ## The use of E will be explained later.
    E = zeros(size(data, 2), size(X, 2), size(X, 2))

    ## Then, We define a empty matrix W with suitable size.

    ## `size(data,2)` and `size(X,2)` are mentioned above.

    ## `size(data,1)` is the number of time slots.

    ## The use of W will also be explained later.
    W = Array{Float64}(undef, size(data, 2), size(X, 2), size(data, 1))

    ## Here we define the progress of our regression, and print out the the number of samples to check the validation of our model
    prog = Progress(size(data, 2) * cross_val_reps; dt = 0.1, enabled = show_progress)
    println("n = samples = $(size(X,1)) = $(size(data,3))")

    ## For each time slot t,
    @showprogress 0.1 for t = 1:size(data, 2)

        ## we call the chosen solver function, with which we do our calculation for as many times as the cross validation defined.

        ## We do this in order to compensate for the reduction in sample size caused by the split (which will be mentioned later)
        for m = 1:cross_val_reps
            k_ix = collect(Kfold(size(data, 3), 2))

            ## According to the BacktoBack algorithm, we need to perform two regressions over the same data sample.

            ## It can result in overfitting, as spurious correlations in the data absorbed by the first regression will be leveraged by the second one.

            ## To avoid this issue, we split our sample (X,Y) into two disjoint sets (X1,Y1) and (X2,Y2).

            ## The first regression will be performed using (X1,Y1), and the second regression will be performed using (X2,Y2).
            Y1 = data[:, t, k_ix[1]]
            Y2 = data[:, t, k_ix[2]]
            X1 = X[k_ix[1], :]
            X2 = X[k_ix[2], :]

            ## The first regression is a backforward regression, which regresses X1 against Y1.

            ## We estimate the linear regression coefficients `G` ̂from Y1 to X1, which recovers the correlations between Y and each factor of X.
            G = model_fun(Y1', X1)

            ## Then, we construct the predictions "X_predicted = YG", and do the second regression from `X2` to X_predicted.

            ## Thus we get the estimation of the linear regression coefficients H from X to X_predicted.
            H = model_fun(X2, Y2'G)

            ## The diagonal of the regression coefficients H is the desired estimate of the causal influence matrix.

            ## We can use the zero matrix E mentioned above for the construction of the causal influence matrix.
            E[t, :, :] = E[t, :, :] + Diagonal(H[diagind(H)])
            ProgressMeter.next!(prog; showvalues = [(:time, t), (:cross_val_rep, m)])
        end

        ## The two successive regressions are repeated over many random splits,

        ## and the final estimate `E` of the causal influence matrix is the average over the estimates associated with each split.
        E[t, :, :] = E[t, :, :] ./ cross_val_reps

        ## The matrix E is then used to estimate the causal influence matrix W,

        ## which is the matrix of regression coefficients from X, which really have influence on Y at time slot t, to Y.
        W[t, :, :] = (X * E[t, :, :])' / data[:, t, :]
    end

    ## Then we extract diagonal,
    beta = mapslices(diag, E, dims = [2, 3])
    ## and reshape to conform to ch x time x pred
    beta = permutedims(beta, [3 1 2])
    modelinfo = Dict("W" => W, "E" => E, "cross_val_reps" => cross_val_reps)
    return Unfold.LinearModelFit(beta, modelinfo)
end;





# ## Modeling function
## The followings are the "key modeling functions" for using our models (according to different regression methods).

## The main idea is to build the model according to the method, then do the tuning if necessary,

## and at last do the calculation and retrive the results.

## Here we have models based on "Least Square", "Ridge", "Lasso", "SVM", and "AdaBoost" regression.
function model_lsq(; kwargs...)
    return (x, y) -> x \ y
end

function model_ridge(; kwargs...)
    model = gen_model_ridge()
    tm = tunemodel(model; kwargs...)
    return (x, y) -> solver_tune(tm, x, y)
end

function model_lasso(; kwargs...)
    model = gen_model_lasso()
    tm = tunemodel(model; kwargs...)
    return (x, y) -> solver_tune(tm, x, y)
end

function model_svm(; kwargs...)
    model = gen_model_svm()
    return (x, y) -> solver_notune(model, x, y)
end

function model_ada(; kwargs...)
    model = gen_model_adaboost()
    return (x, y) -> solver_ada(model, x, y)
end

## The models (except LSQ) are all attached in different packages from Julia.

## To make it explicit, we write the calling of different models from different packages in following functions, which is called by the "key modeling functions" above.
function gen_model_ridge()
    @load RidgeRegressor pkg = MLJLinearModels
    model = MLJLinearModels.RidgeRegressor(fit_intercept = false)
    return model
end

function gen_model_lasso(; kwargs...)
    @load LassoRegressor pkg = MLJLinearModels
    model = MLJLinearModels.LassoRegressor(fit_intercept = false)
    return model
end

function gen_model_svm(; kwargs...)
    @load SVMLinearRegressor pkg = MLJScikitLearnInterface
    model = MLJScikitLearnInterface.SVMLinearRegressor()
    return model
end

function gen_model_adaboost(; kwargs...)
    @load AdaBoostRegressor pkg = MLJScikitLearnInterface
    model = MLJScikitLearnInterface.AdaBoostRegressor()
    return model
end
nothing ## # hide

# ## Tuning function
## For models using "Ridge" and "Lasso", we design a tunemodel for pre-training the hyperparameters, which can improve the performance of regression.

## The function is called in the "key modeling function" above.
function tunemodel(model; nfolds = 5, resolution = 10, measure = MLJ.rms, kwargs...)
    range = Base.range(model, :lambda, lower = 1e-2, upper = 1000, scale = :log10)
    tm = TunedModel(
        model = model,
        resampling = CV(nfolds = nfolds),
        tuning = Grid(resolution = resolution),
        range = range,
        measure = measure,
    )
    return tm
end;

# ## Calculation function
## For different models, we have different parameters, so it would be different for them to calculate the result of the regression.

## (Here we use G as a symbol to represent the result. But actually it can be either G or H defined by our main function)

## They can be splitted into three groups. （LSQ is pretty easy so that it doesn't need a calculation function, so it is not in any group）

## The models using "Ridge" and "Lasso" need tuning, and thus can be in group one (`solver_tune`)

## The model using "SVM" needs no tuning, and thus can be in group two (`solver_notune`)

## The model using "AdaBoost" is kinda tricky. It needs repairation for generating "NaN" values, and thus can be in group three (`solver_ada`)
function solver_tune(tm, data, X)
    G = Array{Float64}(undef, size(data, 2), size(X, 2))
    for pred = 1:size(X, 2)
        mtm = machine(tm, table(data), X[:, pred])
        MLBase.fit!(mtm, verbosity = 0)
        G[:, pred] = Tables.matrix(fitted_params(mtm).best_fitted_params.coefs)[:, 2]
    end
    return G
end

function solver_notune(tm, data, X)
    G = Array{Float64}(undef, size(data, 2), size(X, 2))
    for pred = 1:size(X, 2)
        mtm = machine(tm, table(data), X[:, pred])
        MLBase.fit!(mtm, verbosity = 0)
        G[:, pred] = fitted_params(mtm).coef
    end
    return G
end

function solver_ada(tm, data, X)
    G = Array{Float64}(undef, size(data, 2), size(X, 2))
    for pred = 1:size(X, 2)
        mtm = machine(tm, table(data), X[:, pred])
        MLBase.fit!(mtm, verbosity = 0)
        if isnan(fitted_params(mtm).feature_importances[1])
            G[:, pred] = zeros(20)
        else
            G[:, pred] = fitted_params(mtm).feature_importances
        end
    end
    return G
end
nothing ## # hide

using Unfold
using UnfoldMakie, CairoMakie
using UnfoldSim
using UnfoldDecode
using DataFrames
using Statistics

# ## Data generation

# #### Simulation and data collection
# Collect the data genarated by UnfoldSim, and add certian level of noise
dat, evts = UnfoldSim.predef_eeg(; noiselevel = 0.1, return_epoched = true);
# `dat` is a 45 * 2000 matrix consisting of the result of the dependent variable of the simulation
# `evts` is a matrix consisting of the data of the independent variables during the simulaiton


# #### Renaming
# For better understanding, we have some modifying for our data
# Rename the column `condition` to `animal`
evts = rename(evts,:condition => :animal); 
# Rename the column `continuous` to `eye_angle`
evts = rename(evts,:continuous => :eye_angle); 
# Change the value of the column `animal` to "dog" if the value is "car"
evts.animal[evts.animal .== "car"] .= "dog";
# Change the value of the column `animal` to "cat" if the value is face"
evts.animal[evts.animal .== "face"] .= "cat";
# It doesn't change the values nor the essence of the data, only the way we understand it in the real world


# #### Data further generating
# Add a new column `continuous_random` with random values
evts.continuous_random .= rand(size(evts,1));
# This variable represents the independent variable that doesn't impact the dependent variable
# Add a new column `vegetable` generated according to `eye_angle`
evts.vegetable .= ["tomato","carrot"][1 .+ (evts.eye_angle .+ 10 .* rand(size(evts,1)) .> 7.5)];
# This variable represents the independent variable that can influence the dependent variable, but is correlated to the independent variable `Category`
# Now we have the final version of the Dataframe of the independent variable data needed for our model
#
# cxcccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  add graph ja
#

# #### Dimension expansion
# Repeat the dat 20 times, representing 20 channels and therefore we have a new dimension and permute the dimensions for convenience
dat_3d = permutedims(repeat(dat, 1, 1, 20), [3 1 2]); 
# Add some noise to each channel to better simulate the result in real world
dat_3d .+= 0.1*rand(size(dat_3d)...);
# Now we have the final version of the dependent variable data needed for our model
#
# cxcccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc  add graph nein
#


# ## Building the Model

# #### Solver selection
# Call the solver in UnfoldDecode
# Here we've accomplished link to 3 different methods for regression needed in our algorithm: Ridge, LS, and SVM
# They can be chosen by refering to the parameter `regularization_method`
b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5, regularization_method="RidgeRegressor");
# b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5, regularization_method="LSRegressor")
# b2b_solver = (x, y) -> UnfoldDecode.solver_b2b(x, y; cross_val_reps = 5, regularization_method="SVMLinearRegressor")

# #### Generate the formula
## We build a dataframe which will contain the 4 graphs generated according to the 4 different formulas (which will be mentioned later)
results_all = DataFrame()
for ix = 1:4
    if ix == 1
        f = @formula 0 ~ 1  + animal + eye_angle
    ## The first one takes `animal` and `eye_angle`, which are two independent independent variables that can impact the result into account
    elseif ix == 2
        f = @formula 0 ~ 1  + animal + vegetable 
    ## The second one takes `animal` and `vegetable`, which are two correlated independent variables in which only one variable really affects the result into account
    elseif ix == 3
        f = @formula 0 ~ 1  + animal + vegetable + eye_angle
    ## The third one takes `animal` and `vegetable` and `eye_angle`, which are all variables mentioned above into account
    elseif ix == 4
        f = @formula 0 ~ 1 + animal + eye_angle + continuous_random + vegetable
    ## The last one furtherly takes the randomly generated variable `continuous_random` into account
    end
    ## By comparing the results of the formulas mentioned above, we can see the effect of BacktoBack algorithm
    ## Define a design dictionary according to the formula
    designDict = [Any => (f, range(0, 0.44, step = 1/100))] 
    ## Fit the model
    m = Unfold.fit(UnfoldModel, designDict, evts, dat_3d; solver = b2b_solver)
    ## Present the results in a graph
    results = coeftable(m)
    results.estimate = abs.(results.estimate)
    results = results[results.coefname .!="(Intercept)",:]
    results.formula .= string(f)
    global results_all
    results_all = vcat(results_all,results)
end;

# #### Plot the results
plot_erp(results_all; mapping = (; row = :formula), axis = (xlabel = "Time [s]", ylabel = "Performance"))
#
# We can see from the graph that ......

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
        model = model_ridge()
        tm = tunemodel(model;kwargs...)
## if the chosen regularization_method is `SVMRegressor`, we build a model based on SVMLinearRegressor (without tuning)
    elseif regularization_method == "SVMRegressor"
        tm = model_svm()
    end;
## if the chosen regularization_method is `LSRegressor`, actually we don't need a model here for simplicity

    E = zeros(size(data,2),size(X,2),size(X,2));
    W = Array{Float64}(undef,size(data,2),size(X,2),size(data,1));
    prog = Progress(size(data, 2) * cross_val_reps;dt=0.1,enabled=show_progress);
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

function model_ridge()
    @load RidgeRegressor pkg=MLJLinearModels
    model = MLJLinearModels.RidgeRegressor(fit_intercept=false)
    return model
end;

function model_svm(;kwargs...)
    @load SVMLinearRegressor pkg=MLJScikitLearnInterface # Adjust the package if necessary
    model = MLJScikitLearnInterface.SVMLinearRegressor()
    return model
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


# #### Generate the formula
## We build a dataframe which will contain the 4 graphs generated according to the 4 different formulas (which will be mentioned later)
function run_b2b(f)

    ## By comparing the results of the formulas mentioned above, we can see the effect of BacktoBack algorithm
    ## Define a design dictionary according to the formula
    designDict = [Any => (f, range(0, 0.44, step = 1/100))];
    ## Fit the model
    m = Unfold.fit(UnfoldModel, designDict, evts, dat_3d; solver = b2b_solver);
    ## Present the results in a graph
    results = coeftable(m);
    results.estimate = abs.(results.estimate);
    results = results[results.coefname .!="(Intercept)",:];
    results.formula .= string(f);
    return results
end;

results_all = vcat(run_b2b(@formula 0 ~ 1  + animal + eye_angle),
    ## The first one takes `animal` and `eye_angle`, which are two independent independent variables that can impact the result into account
    ## run_b2b(@formula 0 ~ 1  + animal + vegetable),
    ## The second one takes `animal` and `vegetable`, which are two correlated independent variables in which only one variable really affects the result into account
    run_b2b( @formula 0 ~ 1  + animal + vegetable + eye_angle),
    ## The third one takes `animal` and `vegetable` and `eye_angle`, which are all variables mentioned above into account
    run_b2b(@formula 0 ~ 1 + animal + eye_angle + continuous_random + vegetable));
    ## The last one furtherly takes the randomly generated variable `continuous_random` into account

#
# #### Plot the results
plot_erp(results_all; mapping = (; row = :formula), axis = (xlabel = "Time [s]", ylabel = "Performance"));
#
# We can see from the graph that ......
# Code currently duplicated in Unfold.jl
# https://github.com/unfoldtoolbox/Unfold.jl/edit/main/src/solver.jl

# Basic implementation of https://doi.org/10.1016/j.neuroimage.2020.117028
solver_b2b(X, data, cross_val_reps) = solver_b2b(X, data, cross_val_reps = cross_val_reps)
function solver_b2b(
    X,
    data::AbstractArray{T,3};
    cross_val_reps = 10,
    multithreading = true,
    showprogress=true,
) where {T<:Union{Missing,<:Number}}

    X, data = dropMissingEpochs(X, data)


    E = zeros(size(data, 2), size(X, 2), size(X, 2))
    W = Array{Float64}(undef, size(data, 2), size(X, 2), size(data, 1))

    prog = Progress(size(data, 2) * cross_val_reps, 0.1;enabled=showprogress)
    @maybe_threads multithreading for m = 1:cross_val_reps
        k_ix = collect(Kfold(size(data, 3), 2))
        X1 = @view X[k_ix[1], :]
        X2 = @view X[k_ix[2], :]
            
        for t = 1:size(data, 2)

            Y1 = @view data[:, t, k_ix[1]]
            Y2 = @view data[:, t, k_ix[2]]


            G = (Y1' \ X1)
            H = X2 \ (Y2' * G)

            E[t, :, :] +=  Diagonal(H[diagind(H)])
            ProgressMeter.next!(prog; showvalues = [(:time, t), (:cross_val_rep, m)])
        end
        E[t, :, :] = E[t, :, :] ./ cross_val_reps
        W[t, :, :] = (X * E[t, :, :])' / data[:, t, :]

    end

    # extract diagonal
    beta = mapslices(diag, E, dims = [2, 3])
    # reshape to conform to ch x time x pred
    beta = permutedims(beta, [3 1 2])
    modelinfo = Dict("W" => W, "E" => E, "cross_val_reps" => cross_val_reps) # no history implemented (yet?)
    return LinearModelFit(beta, modelinfo)
end

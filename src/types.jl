struct DecodingFit{T,N} <: Unfold.AbstractModelFit{T,N}
    machines::Vector{MLJ.Machine}
    splits::Tuple
    yhat::AbstractArray
    y::AbstractArray{T,N}
    times::Vector
end

# ╔═╡ cb042a82-f629-4507-b1ca-049e981065ea
struct UnfoldDecodingModel{T} <: UnfoldModel{T}
    design::Vector
    target::Pair
    tbl::DataFrame
    modelfit::Vector{T}
end


function Base.show(io::IO, ::MIME"text/plain", obj::T) where {T<:UnfoldDecodingModel}
    Unfold.Term.tprintln(
        io,
        "Unfold-Type: ::$(typeof(obj)){{$(typeof(obj).parameters[1])}}",
    )
    println(io)

    Unfold.tprint("{gray}Useful functions:{/gray} `coeftable(uf)`")

end

struct DecodingFit <: Unfold.ModelFit
	machines::Vector{MLJ.Machine}
	splits::Tuple
	yhat::Vector
	y::Vector
	times::Vector
end

# ╔═╡ cb042a82-f629-4507-b1ca-049e981065ea
struct UnfoldDecodingModel <: UnfoldModel
	design::Dict
	target::Pair
	tbl::DataFrame
	modelfit::Vector{DecodingFit}
end
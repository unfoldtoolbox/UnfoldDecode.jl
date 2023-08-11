# this should go into Unfold.jl at some point
#  replace https://github.com/unfoldtoolbox/Unfold.jl/blob/2ea1f3cd4e086cfa17b940c95555640d7d1d4228/src/designmatrix.jl#L187C11-L194C16 - designmatrix.jl Line 187

function subsetEvents(tbl,eventname,eventcolumn) 	
	if eventname == Any
		return tbl
	end
	
	  if !((eventcolumn ∈ names(tbl)) | (eventcolumn ∈ propertynames(tbl)))
                error(
                    "Couldnt find columnName: " *
                    string(eventcolumn) *
                    " in event-table.  Maybe need to specify eventcolumn=:correctColumnName (default is ':event') \n names(tbl) = " *
                    join(names(tbl), ","),
                )
	end
	
	         return @view tbl[tbl[:, eventcolumn].==eventname, :] 
end 

"""
`measure`: Any measure of MLJ. Find a whole list: `MLJ.measures()`, by default balanced_accuracy, but root_mean_squared, accuracy etc. all possible.
"""
function Unfold.coeftable(uf::UnfoldDecodingModel;measure=balanced_accuracy,averaged=true)
mf = uf.modelfit
measures = []
for splt = mf
    modes = broadcast(x->broadcast(mode,x),splt.yhat)
    push!(measures,	measure.(modes,Ref(splt.y)))
end



df= DataFrame(
    :coefname => uf.target[2],
    :basisname => uf.target[1],
    :split=> repeat(1:length(uf.modelfit),inner=size(measures[1],1)),
    :time => repeat(mf[1].times,outer=length(measures)),
    :estimate=>vcat(measures...),
)

if averaged
    return groupby(df,Not(:estimate,:split))|>x-> combine(x,:estimate=>mean=>:estimate)
else
    return df
end
#return hcat(measures...)
end
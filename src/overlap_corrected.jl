"""
Returns the single trial corrected (and "epoched") trials from the basis `target`

"""
function singletrials(dat,uf_train::UnfoldLinearModelContinuousTime,
	tbltrain::DataFrame,target,eventcolumn)
  
	basis = Unfold.design(uf_train)[target][2]
		
		
	eventTbl = subsetEvents(tbltrain,target,eventcolumn)

	
	onsets = eventTbl.latency
	startIX = floor.(onsets) .+ Unfold.shiftOnset(basis)
	stopIX = startIX .+ length(Unfold.times(basis)).-1
	
	
	X_train = modelmatrix(uf_train)

	# get predicted singler trial events
	# XXX multi-basis, this will select the wrong formula

	#@assert(target == Any, "multibasis not yet implemented, probably here the only problem could occur!")
	forms =  Unfold.formula(uf_train)
	if forms isa Array
		# find out which basis to use for decoding
		function whichevent(uf,target,eventcolumn)
			for (ix,e) = enumerate(Unfold.designmatrix(uf).events)
				if e[1,eventcolumn] == target
					return ix
				end
			end
		end
		ix = whichevent(uf_train,target,eventcolumn)
		
	else
		forms = [forms.rhs]
		ix = 1
	end
	
	fromTo,timesvec,eff =  Unfold.yhat(uf_train,forms,eventTbl)
	if forms isa Array
		
		mxsum = cumsum(maximum.(fromTo))
		
		
		fromTo = fromTo[ix] 
		if ix>1
			fromTo .+= mxsum[ix-1]
		end
	else
		fromTo = fromTo[1]
	end

	
	out_e = Array{Float64}(undef,size(dat,1),fromTo.step,length(fromTo))
	perevent_views = view.(Ref(eff),fromTo,Ref(1:size(eff,2)))

	for e = 1:length(onsets)
	
		# index
		ix = startIX[e]:(stopIX[e]-1)
		# clip index, necessary for out_e
		ixix = (ix.>0) .&& (ix .< size(dat,2))
	
		# index into data / designmatrix
		dataIx = ix[ixix]
		# calculate residuals
		resid =   @view(dat[:,dataIx]) .- (@view(X_train[dataIx,:]) * coef(uf_train)')'
		# add back the current response, now overlap corrected :)
		ix_eff = (fromTo[e]:fromTo[e]+fromTo.step-1)
		
		out_e[:,ixix,e] .=resid  .+ @view(eff[ix_eff[ixix],:])'
	end
	return out_e
end

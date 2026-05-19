# weights.jl

function weights(p;rowsum=true)

    wgt_distribution = Normal(0, p.sigma/sqrt(p.Ncells))

    nc0Max = p.Ncells # outdegree = Ncells - 1
    w0Weights = zeros(nc0Max, p.Ncells)   # postcell x precell
    for i = 1:p.Ncells
        w0Weights[:,i] .= rand(wgt_distribution, nc0Max) # gaussian weights
    end
    if rowsum
        return w0Weights .- mean(w0Weights,dims=2)
    else
        return w0Weights
    end

end

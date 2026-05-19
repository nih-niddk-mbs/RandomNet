
# measures.jl


"""
phase_indices(phase,Nphases,u,l)

Assign index on domain 1 to Nphases to phase on domain l to u

"""
phase_index(phase, Nphases, lower=-pi, upper=pi) = max(ceil(Int, (phase - lower) / (upper - lower) * Nphases), 1)
phase_indices(phases, Nphases, lower=-pi, upper=pi) = max.(ceil.(Int, (phases .- lower) / (upper - lower) * Nphases), 1)


"""
rho(phases::Matrix,Nphases)
rho(phases::Vector,Nphases)


"""

mean_a3(phases::Matrix, Nphases, domain=2pi) = mean(compute_a3(phases, Nphases, domain), dims=2)[:, 1]

function compute_a3(phases::Matrix, Nphases, domain=2pi)
    rho = zeros(Nphases, size(phases, 2))
    compute_a3!(rho, phases, Nphases, domain)
    return rho
end

function compute_a3(phases::Vector, Nphases, domain=2pi)
    rho = zeros(Nphases)
    compute_a3!(rho, phases, Nphases, domain)
    return rho
end
"""
rho!(rho,phases,Nphases)

Find mean phase density
averaged over network and time

- `phase': matrix of phases where rows = neurons, cols = time

"""
function compute_a3!(rho::Matrix, phases, Nphases, domain)
    dN = 1 / size(phases, 1) * Nphases / domain
    for t in axes(phases, 2)
        phase_idx = phase_indices(phases[:, t], Nphases)
        for i in phase_idx
            rho[i, t] += dN
        end
    end
end
function compute_a3!(rho::Vector, phases, Nphases, domain)
    phase_idx = phase_indices(phases, Nphases)
    dN = 1 / length(phases) * Nphases / domain
    for i in phase_idx
        rho[i] += dN
    end
end

"""
covariance(x,y,lags)

Find cross-covariance function of matrices x and y over time lags
covariance is taken over neurons, while function is averaged over time
- `x`,`y`: are matrices where rows are neuron indices and columns are time indices

"""
function covariance(x, y, lags)
    T = size(x, 2) - lags[end]
    c = zeros(length(lags))
    for t in 1:T
        for (i, lag) in enumerate(lags)
            c[i] += cov(x[:, t], y[:, t+lag])
        end
    end
    c / T
end


"""
c11(u,lags)

Find covariance function of synaptic input

- `u`: matrix of inputs, rows = neurons, cols = time
"""
compute_C11(u, lags) = covariance(u, u, lags)

"""
m13(phases,u,Nphases)

"""
function compute_M13(phases::Vector, u::Vector, Nphases, domain=2pi)
    m = zeros(Nphases)
    dN = 1 / size(phases, 1) * Nphases / domain
    phase_idx = phase_indices(phases, Nphases)
    for (i, p) in enumerate(phase_idx)
        m[p] += u[i] * dN
    end
    m
end
"""
    c13(phases,u,Nphases)

c13(phases,u,Nphases)

"""
compute_C13(phases, u, Nphases) = compute_M13(phases, u, Nphases) - mean(u) * compute_a3(phases, Nphases)

function compute_C13(phases, u, lags, Nphases, domain=2pi)
    T = size(phases, 2) - lags[end]
    c = zeros(Nphases, length(lags))
    a1 = mean(u, dims=1)
    for t in 1:T
        a3 = compute_a3(phases[:, t], Nphases, domain)
        for (ind, lag) in enumerate(lags)
            c[:, ind] .+= compute_M13(phases[:, t], u[:, t+lag], Nphases) - a3 * a1[t+lag]
        end
    end
    c / T
end

"""
compute_M33(idx1,idx2,Nphases,domain=2pi)

"""
function compute_M33(idx1, idx2, Nphases, domain=2pi)
    c = zeros(Nphases, Nphases)
    dN = (Nphases / domain)^2 / length(idx1)
    for i in eachindex(idx1)
        c[idx1[i], idx2[i]] += dN
    end
    c
end
"""
compute_C33(phases,lags,Nphases,domain=2pi)


"""

function compute_C33(phases, lags, Nphases, domain=2pi)
    T = size(phases, 2) - lags[end]
    c = zeros(Nphases, Nphases, length(lags))
    rho = compute_a3(phases, Nphases, domain)
    phase_idx = phase_indices(phases, Nphases)
    for t in 1:T
        for (ind, lag) in enumerate(lags)
            c[:, :, ind] .+= compute_M33(phase_idx[:, t], phase_idx[:, t+lag], Nphases) .- rho[:, t] * rho[:, t+lag]'
        end
    end
    c / T
end
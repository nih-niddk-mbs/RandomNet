# integrator.jl


function integrate(p, W, f, sample_neurons=100)
    v = p.reset * rand(p.Ncells) .+ (p.threshold - p.reset) #membrane voltage
    u = zeros(p.Ncells)
    integrate(p, W, f, u, v, sample_neurons)
end

ftheta(u, v, I, alpha) = 1 - cos(v) + (I + u) * (1 + cos(v))
fphase(u, v, I, alpha) = alpha * (I + u)^(1 / alpha)
flif(u, v, I, alpha) = I + u - alpha * v

function integrate(p, W, f, u::Vector, v::Vector, sample_neurons=100)

    vt = p.threshold # firing threshold
    vr = p.reset  # reset strength
    dt = p.dt # time step size
    Nsteps = p.Nsteps  # simulation steps
    Ncells = p.Ncells # number of neurons
    extInput = p.Input # external input
    beta = p.beta # synaptic time
    alpha = p.alpha # auxilliary neuron parameter

    maxTimes = round(Int, p.maxrate * Nsteps * dt / 1000) # max number of save spike times
    times = zeros(Ncells, maxTimes)  # spike times
    ns = zeros(Int, Ncells)   # index for which neuron spiked

    Inputs = zeros(Ncells) #summed weight of incoming spikes
    InputsPrev = zeros(Ncells) #as above, for previous timestep

    vTotal = zeros(sample_neurons, Nsteps)
    uTotal = zeros(sample_neurons, Nsteps)

    for ti in 1:Nsteps
        if mod(ti, Nsteps / 100) == 1  #print percent complete
            print("\r", round(Int, 100 * ti / Nsteps))
        end

        t = dt * ti
        Inputs .= 0.0

        for ci in 1:Ncells
            u[ci] += -dt * beta * u[ci] + InputsPrev[ci] * beta
            vel = f(u[ci], v[ci], extInput, alpha)  # v velocity
            v[ci] += dt * vel

            if v[ci] > vt  #spike occurred
                discount = exp(beta * (vt - v[ci]) / vel)  # discount input by amount of time since crossing threshold
                v[ci] -= vr
                ns[ci] += 1
                if ns[ci] <= maxTimes
                    times[ci, ns[ci]] = t
                end
                for j = 1:Ncells
                    Inputs[j] += W[j, ci] * discount
                end #end loop over synaptic projections
            end #end if(spike occurred)
            # saved for visualization
            if ci <= sample_neurons
                vTotal[ci, ti] = v[ci]
                uTotal[ci, ti] = u[ci]
            end

        end #end loop over neurons

        InputsPrev = copy(Inputs)

    end #end loop over time
    print("\r")

    return times, ns, uTotal, vTotal, u, v


end

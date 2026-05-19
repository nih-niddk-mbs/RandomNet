module RandomNet


using Distributions
using Plots
using LinearAlgebra
# using Random
# using Smoothers

export
    plot,
    set_params,
    make_raster,
    make_phaseplot,
    make_synplot,
    weights,
    integrate,
    compute_rho,
    mean_rho,
    compute_C11,
    compute_C13,
    compute_C33,
    steady_state,
    ftheta,
    fphase,
    flif



include("weights.jl")
include("integrator.jl")
include("measures.jl")
include("2PI.jl")

"""
phaseparams


"""
struct phaseparams
    dt::Float64
    Nsteps::Int64
    Ncells::Int64
    maxrate::Float64
    sigma::Float64
    Input::Float64
    threshold::Float64
    reset::Float64
    beta::Float64
    alpha::Float64
end


"""


"""
function set_params(;
    dt=0.1, #simulation timestep (ms)
    Nsteps=20000, # total time of simulation
    Ncells=10000, # number of neurons in network
    maxrate=500, # maximum average firing rate.
    sigma=0.2, # connection weight std
    Input=0.005,  # external input
    threshold=pi,
    reset=2pi,
    beta=0.1, #40 #synaptic drive decay rate
    alpha=0.5 # extra parameter
)

    phaseparams(dt, Nsteps, Ncells, maxrate, sigma, Input, threshold, reset, beta, alpha)

end

function make_raster(times, ns, Nraster=100)
    # raster plot
    figure(figsize=(10, 10))
    for ci = 1:Nraster
        plot(times[ci, 1:ns[ci]], ci * ones(ns[ci]), marker=".", linestyle="", color="red")
    end
    ylim([0, Nraster])
    tight_layout()
end

function make_phaseplot(timev, phase)
    # membrane potential
    figure(figsize=(8, 4))
    ci = 1
    plot(timev, phase[tind, ci], linewidth=1.0, color="C$(ci)", alpha=1.0)
    xlabel("time", fontsize=15)
    ylabel("phase", fontsize=15)
    tight_layout()
end


function make_synapseplot(synInTotal)
    # synaptic currents
    figure(figsize=(6, 4.5))
    ci = 1
    # plot(timev,synInExc[tind,ci] .+ p.muemax,color="red",linewidth=0.5,alpha=1.0,label="synIn_exc + ext")
    # plot(timev,synInInh[tind,ci],color="blue",linewidth=0.5,alpha=2.0,label="synIn_inh")
    plot(timev, synInTotal[tind, ci], color="black", linewidth=0.5, alpha=1.0, label="synIn_total")
    legend(fontsize=10, frameon=false)
    xlabel("time", fontsize=15)
    ylabel("u", fontsize=15)
    tight_layout()
end


end # module


# tind = collect(1:pars.Nsteps)
# timev = pars.dt*tind
#
# W = Weights(pars)
#
# _, _, _, _, v0, u0 = run(pars,W)
# times, ns, phases, synInTotal, v0, u0 = run(pars,W,v0,u0)
#
# tind = collect(1:pars.Nsteps)
# timev = pars.dt*tind
#
#
# make_raster(times,ns)

#----- autocovariance (neuron) -----#
# lags = collect(0:100:10000)
# neuronAutocorr = autocorNeuron(p, times, lags)
# neuronavgAutocorr = mean(neuronAutocorr, dims=1)[:]
# popAutocorr = autocorPop(p, times, lags)
# c13 = C13(p,phase,u)


# autocor
# figure(figsize=(10,15))
# subplot(211)
# plot(p.dt*lags[1:end], popAutocorr[1:end], label="pop")
# legend()
# subplot(212)
# plot(p.dt*lags[1:end], neuronavgAutocorr[1:end], label="neuron")
# # title("std: $(round(std(avgAutocorr[500:2000]),digits=7))")
# legend()
# tight_layout()


# # inter spike interval
# cv_isi = zeros(p.Ncells)
# for ci = 1:p.Ncells
#   global cv_isi
#   nsmin = minimum([ns[ci] size(times)[2]])
#   isi_ci = times[ci,2:nsmin] - times[ci,1:nsmin-1]
#   cv_ci = std(isi_ci)/mean(isi_ci)
#   if isnan(cv_ci) == false
#     cv_isi[ci] = cv_ci
#   end
# end
# cv_isi_nonzero = cv_isi[cv_isi .> 0]
# cv_isi_mean = mean(cv_isi_nonzero)
# println("CV of ISI: ",cv_isi_mean)

# # CV distribution
# cv_isi_exc = cv_isi[1:p.Ne]
# cv_isi_exc = cv_isi_exc[cv_isi_exc .> 0]
# figure(figsize=(4,3.5))
# hist(cv_isi_nonzero,bins=100,range=(0,2),color="k",histtype="step")
# # hist(cv_isi_exc,bins=100,range=(0,2),color="k",histtype="step")
# title("mean CV = $(round(cv_isi_mean,digits=2))",fontsize=15)
# xlabel("CV of all neurons",fontsize=15)
# tight_layout()


# # rate vs CV
# rates = ns/(p.train_time/1000)
# figure(figsize=(4,3.5))
# plot(rates, cv_isi, marker="o", markersize=5, markerfacecolor="none", linestyle="", color="black", alpha=0.3)
# # plot(rates[1:p.Ne], cv_isi[1:p.Ne], marker="o", markersize=5, markerfacecolor="none", linestyle="", color="black", alpha=0.3)
# xlim([0, 40])
# ylim([0, 2.5])
# xlabel("rates (Hz)", fontsize=15)
# ylabel("CV", fontsize=15)
# tight_layout()


# # log-normal rates
# rates = ns/(p.train_time/1000)
# figure(figsize=(4,3.5))
# # hist(log10.(rates[p.Ne+1:p.Ncells]), bins=100, width=0.05, range=(-3,3), density=true)
# hist(log10.(rates), bins=100, width=0.05, range=(-3,3), density=true, color="black")
# # ylim([0, 1])
# xlabel("log10 of rates (Hz)", fontsize=15)
# ylabel("pdf", fontsize=15)
# tight_layout()


# # population rate
# excrate = mean(ns[1:p.Ne])/(p.train_time/1000)
# inhrate = mean(ns[p.Ne+1:p.Ncells])/(p.train_time/1000)
# println("excitatory rate: ", round(excrate,digits=2), " Hz")
# println("inhibitory rate: ", round(inhrate,digits=2), " Hz")

# qe, qi = p.Ne/p.Ncells, p.Ni/p.Ncells
# J = [p.jee*p.sqrtK*qe/p.taue p.jei*p.sqrtK*qi/p.taue; p.jie*p.sqrtK*qe/p.taui p.jii*p.sqrtK*qi/p.taui]
# X = [p.muemax/p.sqrtK/p.taue; p.muimax/p.sqrtK/p.taui]
# R = -inv(J)*X
# println("predicted excitatory rate: ",round(R[1]*1000,digits=2), " Hz")
# println("predicted inhibitory rate: ",round(R[2]*1000,digits=2), " Hz")

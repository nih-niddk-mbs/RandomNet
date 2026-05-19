### 2PI.jl

# Functions for theta neuron, v= original coordinates
F(v, I) = 1 - cos(v) + I * (1 + cos(v))
Fu(v) = 1 + cos(v)
Fv(v, I) = (1 - I) * sin(v)
F_phi(phi, I) = 4I / (1 + cos(phi) + I * (1 - cos(phi)))
Fu_phi(phi) = 2 * (1 + cos(phi)) / (1 + cos(phi) + I * (1 - cos(phi)))
Fv_phi(phi, I) = (1 - I) * I * sin(phi) / (1 + cos(phi) + I * (1 - cos(phi)))

# flux derivatives for 2PI equations
dF(a, a1, I, phi, phi1, dphi) = (F(phi, I) * a - F(phi1, I) * a1) / dphi
dFu(a, a1, phi, phi1, dphi) = (Fu(phi) * a - Fu(phi1) * a1) / dphi
dFvu(a, a1, v, v1, dphi) = (Fvu(v) * a - Fvu(v1) * a1) / dphi

"""
make_phases(N,domain=2pi) = collect(1:N) * domain/N .- .5*domain

Make vector of phases of length N on domain -domain/2 to domain/2

"""
make_phases(N, domain=2pi) = collect(1:N) * domain / N .- 0.5 * domain

"""
phase_set(a,domain=2pi)
phase_set(N::Int,domain=2pi)

"""

phase_set(a, domain=2pi) = phase_set(size(a, 1), domain)

function phase_set(N::Int, domain=2pi)
    phi = make_phases(N, domain)
    dphi = domain / N
    return phi, dphi, N
end




"""
transform(phi::Float64,I) = 2*atan(sqrt(I)*tan(phi/2))
transform_inv(v::Float64,I) =2*atan(tan(v/2)/sqrt(I))

Transform to characteristic coordinates and inverse (i.e. from v to phi and back)
"""
transform(phi::Float64, I) = 2 * atan(sqrt(I) * tan(phi / 2))
transform_inv(v::Float64, I) = 2 * atan(tan(v / 2) / sqrt(I))

function transform(a::Vector, I, f=transform)
    N = length(a)
    dphi = 2pi / N
    phi = make_phases(N)
    at = similar(a)
    for i in 1:N
        x = f(phi[i], I)
        ix = phase_index(x, N)
        ix1 = ix == 1 ? N : ix - 1
        dp = phi[ix] - x
        at[i] = ((dphi - dp) * a[ix] + dp * a[ix1]) / dphi
    end
    return at
end


function transform(M::Matrix, phi, I, f=transform)
    N = size(M, 2)
    dphi = 2pi / N
    Mt = similar(M)
    for i in 1:N
        for j in 1:N
            x = f(phi[i], I)
            y = f(phi[j], I)
            ix = phase_index(x, N)
            iy = phase_index(y, N)
            ix1 = ix == 1 ? N : ix - 1
            iy1 = iy == 1 ? N : iy - 1
            dx = phi[ix] - x
            dy = phi[iy] - y
            Mt[i, j] = ((dphi - dy) * ((dphi - dx) * M[ix, iy] + dx * M[ix1, iy]) + dy * ((dphi - dx) * M[ix, iy1] + dx * M[ix1, iy1])) / dphi^2
        end
    end
    return Mt
end

function transform_inv(M::Matrix, phi, I)
    N = size(M, 2)
    dphi = 2pi / N
    Mt = similar(M)
    for i in 1:N
        for j in 1:N
            x = transform_inv(phi[i], I)
            y = transform_inv(phi[j], I)
            ix = phase_index(x, N)
            iy = phase_index(y, N)
            dx = phi[ix] - x
            dy = phi[iy] - y
            Mt[i, j] = ((dphi - dy) * ((dphi - dx) * M[ix, iy] + dx * M[ix-1, iy]) + dy * ((dphi - dx) * M[ix, iy-1] + dx * M[ix-1, iy-1])) / dphi^2
        end
    end
    return Mt
end

jacobian(phi, I) = 2 * sqrt(I) / (1 + cos(phi) + I * (1 - cos(phi)))

function make_dFa(a3, df=dFu)
    phi, dphi, N = phase_set(a3)
    dFa = zeros(N)
    for i in 1:N
        i1, im1 = index_sym(i, N)
        dFa[i] = df(a3[i1], a3[im1], phi[i1], phi[im1], 2 * dphi)
    end
    return dFa
end

function make_dFadFa(a3, phi, N, df=dFu)
    dphi = 2pi / N
    dFadFa = zeros(N, N)
    for i in 1:N
        for j in 1:N
            i1, i2 = index_sym(i, N)
            j1, j2 = index_sym(j, N)
            dFadFa[i, j] = df(a3[i1], a3[i2], phi[i1], phi[i2], 2 * dphi) * df(a3[j1], a3[j2], phi[j1], phi[j2], 2 * dphi)
        end
    end
    return dFadFa
end


"""
smooth_a3(v::Matrix,N,Input)

Create phi transformed and smoothed a3 from simulation phase data (using moving average smoothing)

"""
function smooth_a3(v::Matrix, N, Input)
    a3 = mean_a3(v, N + 9)
    a3phi = transform(a3, Input)
    sma(a3phi, 10)
end

function smooth_C33(v::Matrix, N, Input)
    phi = make_phases(N)
    C33 = compute_C33(v, collect(1:2), N)
    for i in 1:2
        C33[:, :, i] = transform(C33[:, :, i], phi, Input)
    end
    return C33, phi
end
function compute_D33(C33, Input)
    phi, _, N = phase_set(C33)
    D33 = Matrix{Float64}(undef, N, N)
    for i in 1:N, j in 1:N
        ip = i == N ? 1 : i + 1
        D33[i, j] = C33[ip, j, 2] - C33[i, j, 1] * exp((F_phi(phi[ip], Input) - F_phi(phi[i], Input)) / 2 / sqrt(Input))
    end
    return D33 * N * sqrt(Input) / pi
end

function compute_D33a(C33, Input, fprime=Fv)
    phi, _, N = phase_set(C33)
    D33 = Matrix{Float64}(undef, N, N)
    for i in 1:N, j in 1:N
        ip = i == N ? 1 : i + 1
        D33[i, j] = (C33[i, j, 2] - C33[i, j, 1] + C33[ip, j, 2] - C33[i, j, 2]) * N * sqrt(Input) / pi + fprime(phi[i], Input) * C33[i, j, 1]
    end
    return D33
end

steadystate_a30(I, phi, f) = sqrt(I) / pi ./ f.(phi, I)


function steadystate_a3(a3, C13, I, N)
    dphi = 2pi / N
    phi = collect(1:N) * dphi .- pi .- 0.5 * dphi
    a30 = 1 ./ F.(phi, I)
    A = Fu.(phi) ./ F.(phi, I) .* C13
    K = (1 + sum(A) * dphi) / (sum(a30) * dphi)
    return K * a30 .- A, F.(phi, I) .* a3 + Fu.(phi) .* C13 .- K, a30 * sqrt(I) / pi
end

"""
    step_a3!(a3,C31,I,sigma2,h,N,phi,dphi)


"""


# function step_a3!(a3,C31,I,sigma2,h,N,dphi)
#     for i in 1:N
#         i1 = i == 1 ? N : i - 1
#         a3[i] -= h*(dF(a3[i],a3[i1],I,phi[i],phi[i1],dphi) - sigma2*dFu(C31[i],C31[i1],phi[i],phi[i1],dphi))
#     end
# end

# function step_a3!(a3,C31,I,sigma2,h,N,dphi,T)
#     for t in 1:T
#         step_a3!(a3,C31,I,sigma2,h,N,dphi)
#     end
# end

# function step_D33!(D33,C11,dFadFa,Input,h,N,phi)
#     nu = 2*sqrt(Input)
#     for i in 1:N
#         for j in 1:N
#             jp = j == N ? 1 : j + 1
#             D33[i,j] = -h*dFadFa[i,jp]*exp((F_phi(phi[jp],I)-F_phi(phi[j],Input))/nu)*C11 + D33[i,jp]*exp((F_phi(phi[jp],Input)-F_phi(phi[j],Input))/nu)
#         end
#     end
# end

# function step_C33!(C33,D33,Input,h,N,phi)
#     nu = 2*sqrt(Input)
#     for i in N:-1:1
#         in = i == 1 ? N : i - 1
#         for j in 1:N
#             C33[i,j] = (h*D33[in,j] + C33[in,j])*exp((F_phi(phi[in],Input)-F_phi(phi[i],Input))/nu)
#         end
#     end
# end

step_C11(C11, D11, h) = C11 + h * D11

step_D11(D11, C11, C33, h, beta2, sig2) = D11 + h * beta2 * (C11 - 4 * sig2 * C33)



# function evolve_C33(C11,D33in,a3,Input,T,lag=1)
#     phi,dphi,N = phase_set(a3)
#     dFadFa = make_dFadFa(a3,phi,N)
#     nu = 2*sqrt(Input)
#     C33 = diagm(a3)/dphi - a3*a3'
#     D33 = copy(D33in)
#     h = 2pi/N/nu
#     k = 1
#     C33tot = zeros(N,N,T+1)
#     D33tot = zeros(N,N,T+1)
#     C33tot[:,:,k] = C33
#     D33tot[:,:,k] = D33
#     for t in 1:T
#         step_D33!(D33,C11[t],dFadFa,Input,h,N,phi)
#         step_C33!(C33,D33,Input,h,N,phi)
#         if mod(t,lag) == 0
#             k += 1
#             C33tot[:,:,k] = C33
#             D33tot[:,:,k] = D33
#         end
#     end
#     return C33tot,D33tot,C33,D33
# end

# function evolve(C11,D33in,a3,p,T,lag=1)
#     Input = p.Input
#     sig2 = p.sigma^2
#     beta2 = p.beta^2
#     phi,dphi,N = phase_set(a3)
#     dFadFa = make_dFadFa(a3,phi,N)
#     nu = 2*sqrt(Input)
#     D33 = copy(D33in)
#     C33 = diagm(a3)/dphi - a3*a3'
#     h = 2pi/N/nu
#     k = 1
#     D11 = 0
#     C11tot = zeros(T+1)
#     C33tot = zeros(N,N,T+1)
#     D33tot = zeros(N,N,T+1)
#     C33tot[:,:,k] = C33
#     D33tot[:,:,k] = D33
#     C11tot[k] = C11
#     for t in 1:T
#         D11 = step_D11(D11,C11,C33[end,end],h,beta2,sig2)
#         C11 = step_C11(C11,D11,h)
#         step_D33!(D33,C11,dFadFa,Input,h,N,phi)
#         step_C33!(C33,D33,Input,h,N,phi)
#         if mod(t,lag) == 0
#             k += 1
#             C33tot[:,:,k] = C33
#             D33tot[:,:,k] = D33
#             C11tot[k] = C11
#         end
#     end
#     return C33tot,D33tot,C11tot,C33,D33,C11
# end


function advect!(a, h, T)
    for t in 1:T
        advectfd!(a, h)
    end
end

function advectright!(a, Input, h, phi, dphi, N)
    a0 = copy(a)
    for i in 1:N
        i1 = i == 1 ? N : i - 1
        a[i] = a0[i] - h * (dF(a0[i], a0[i1], Input, phi[i], phi[i1], dphi))
    end
end

function advectleft!(a, Input, h, phi, dphi, N)
    a0 = copy(a)
    for i in 1:N
        i1 = i == N ? 1 : i + 1
        a[i] = a0[i] - h * (dF(a0[i], a0[i1], Input, phi[i], phi[i1], dphi))
    end
end

function evolve_a(ain, Input, T, h, (f!)=advectleft!)
    a = copy(ain)
    phi, dphi, N = phase_set(a)
    atot = zeros(N, T + 1)
    atot[:, 1] = a
    k = 2
    for t in 1:T
        f!(a, Input, h, phi, dphi, N)
        atot[:, k] = a
        k += 1
    end
    return atot, a
end

# function indexr(i,j,N)
#     if i == j
#         if i == 1
#             i1 = N
#             j1 = j
#         else
#             i1 = i
#             j1 = j-1
#         end
#     else
#         i1 = i - 1
#         j1 = j
#     end
#     i1,j1
# end

# function indexl(i,j,N)
#     if j == i
#         if j == N
#             i1 = i
#             j1 = 1
#         else
#             i1 = i+1
#             j1 = j
#         end
#     else
#         i1 = i
#         j1 = j + 1
#     end
#     i1,j1
# end

indexp1(i, N) = i == N ? 1 : i + 1
indexm1(i, N) = i == 1 ? N : i - 1

function index_sym(i, N)
    if i == 1
        i1 = i + 1
        im1 = N
    else
        if i == N
            i1 = 1
        else
            i1 = i + 1
        end
        im1 = i - 1
    end
    i1, im1
end


function stepfd_C33!(C33, D33, Input, h, N, phi, dphi)
    C330 = copy(C33)
    for i in 1:N
        for j in 1:N
            ir = indexm1(i, N)
            C33[i, j] = C330[i, j] - h * (dF(C330[i, j], C330[ir, j], Input, phi[i], phi[ir], 2 * dphi) - D33[i, j])
        end
    end
end

function stepfd_D33!(D33, C11, dFadFa, Input, h, N, phi, dphi)
    D330 = copy(D33)
    for i in 1:N
        for j in 1:N
            jl = indexp1(j, N)
            D33[i, j] = D330[i, j] - h * (dF(D330[i, j], D330[i, jl], Input, phi[j], phi[jl], 2 * dphi) + dFadFa[i, j] * C11)
        end
    end
end

function evolvefd_C33(C11, a3, I, h, T, lag=1)
    phi, dphi, N = phase_set(a3)
    dFadFa = make_dFadFa(a3, phi, N)
    C33 = diagm(a3) / dphi - a3 * a3'
    D33 = zeros(N, N)
    evolvefd_C33(C33, D33, C11, a3, I, h, T, lag)
end


function evolvefd_C33(C33, D33, C11, a3, Input, h, T, lag=1)
    phi, dphi, N = phase_set(a3)
    dFadFa = make_dFadFa(a3, phi, N)
    C33tot = zeros(N, N, div(T, lag) + 1)
    C33tot[:, :, 1] = C33
    k = 2
    for t in 1:T
        stepfd_C33!(C33, D33, I, h, N, phi, dphi)
        stepfd_D33!(D33, C11, dFadFa, Input, h, N, phi, dphi)
        if mod(t, lag) == 0
            C33tot[:, :, k] = C33
            k += 1
        end
    end
    return C33tot, C33, D33
end


"""
    evolvefd(C11,D33,a3,p,h,T,lag=1)
    evolvefd(C33in,D33in,C11,D11,a3,p,h,T,lag=1)

TBW
"""
evolvefd(D33, C11, a3, p, h, T, lag=1) = evolvefd(diagm(a3) * length(a3) / 2pi - a3 * a3', D33, C11, 0, a3, p, h, T, lag)


function evolvefd(C33in, D33in, C11, D11, a3, p, h, T, lag=1)
    Input = p.Input
    sig2 = p.sigma^2
    beta2 = p.beta^2
    phi, dphi, N = phase_set(a3)
    dFadFa = make_dFadFa(a3, phi, N)
    D33 = copy(D33in)
    C33 = copy(C33in)
    k = 1
    C11tot = zeros(T + 1)
    C33tot = zeros(N, N, T + 1)
    D33tot = zeros(N, N, T + 1)
    C33tot[:, :, k] = C33
    D33tot[:, :, k] = D33
    C11tot[k] = C11
    for t in 1:T
        D11 = step_D11(D11, C11, C33[end, end], h, beta2, sig2)
        C11 = step_C11(C11, D11, h)
        stepfd_D33!(D33, C11, dFadFa, Input, h, N, phi, dphi)
        stepfd_C33!(C33, D33, Input, h, N, phi, dphi)
        if mod(t, lag) == 0
            k += 1
            C33tot[:, :, k] = C33
            D33tot[:, :, k] = D33
            C11tot[k] = C11
        end
    end
    return C33tot, D33tot, C11tot, C33, D33, C11, D11
end

"""
    step_C33tt(C33,C31,a3,h,N,dphi)

TBW
"""

"""
    steady_state()

TBW
"""

# function steady_state(C11,T,N,I,beta,sigma,h,iterations)
#     sigma2 = sigma^2
#     a3,C32,C31,D31,C33 = initialize(C11,T,N,I,beta,h)
#     steady_state!(a3,C11,C31,D31,C32,C33,I,beta,sigma2,h,N,2pi/N,T,iterations)
# end
# function steady_state!(a3,C11,C31,D31,C32,C33,I,beta,sigma2,h,N,dphi,T,iterations)
#     for i in 1:iterations
#         step_C31tt!(C31,a3,C11,D31,I,beta,h,N,dphi)
#         step_a3!(a3,C31,I,sigma2,h,N,dphi,T)
#         # step_C33tt!(C33,C31,a3,I,h,N,dphi,T)
#         # step_C33!(C33,C31,a3,I,h,N,dphi,T)
#         # step_C32!(C32,a3,I,t,beta,h,N,dphi)
#         # D31 = update_D31(4*sigma2*C33,C32,lag)
#     end
#     return a3,C31,D31,C32,C33
# end

# function initialize(C11,T,N,I,beta,h)
#     dphi = 2pi/N
#     phi = dphi*collect(1:N) .- pi
#     a3 = 1 ./ F.(phi,I)
#     a3 /= sum(a3)
#     C31 = step_C31tt(a3,C11,I,beta,h,N,phi,dphi)
#     # C33 = step_C33(C31,a3,I,h,N,T)
#     # C32 = step_C32(a3,I,beta,h,N,dphi,T)
#     # D31 = update_D31(4*sigma*2*C33,C32,0)
#     C32 = zeros(T)
#     D31 = zeros(N)
#     C33 = zeros(N,N)
#     return a3,C32,C31,D31,C33
# end

using LinearAlgebra
using Plots
using BenchmarkTools
using LaTeXStrings
using NPZ






#optomech-dynamics in rotating frame under driving and homodyne detection.
function dynamics_matrix(
    u::Array{Float64},
    signal::Array{Float64},
    dynamics_parameters::Array{Float64},
    ) #
    γ, η, Γ, n = dynamics_parameters
    ζ = 4.0 * η * Γ
    #X,Y Vx Vy, C = u;
    #dX = -((0.5 * γ1) * X + 4.0 * η1 * Γ1 * (Vx * X + C * Y) + drive[1]) * dt +sqrt(4.0 * η1 * Γ1) * (Vx * signal[1] + C * signal[2])

    du1 =
        -(0.5 * γ + ζ * u[3]) * u[1] - ζ * u[5] * u[2]  +
        sqrt(ζ) * (u[3] * signal[1] + u[5] * signal[2])
    du2 =
        -(0.5 * γ + ζ * u[4]) * u[2] - ζ * u[5] * u[1] +
        sqrt(ζ) * (u[4] * signal[2] + u[5] * signal[1])
    du3 = γ * (n + 0.5 - u[3]) + Γ - ζ * (u[3] * u[3] + u[5] * u[5]) # σxx
    du4 = γ * (n + 0.5 - u[4]) + Γ - ζ * (u[4] * u[4] + u[5] * u[5]) # σyy
    du5 = -γ * u[5] - ζ * (u[3] + u[4]) * u[5] # σxy
    dl = sqrt(ζ) * (u[1] * signal[1] + u[2] * signal[2]) - 0.5 * ζ *(u[1]^2.0 + u[2]^2.0 ) #loglike
    #dl = sqrt(ζ) * (u[1] * signal[1] ) - 0.5 * ζ *(u[1]^2.0) #loglike

    return [du1, du2, du3, du4, du5, dl]
end

#homodyne current
function signal_matrix(u::Array{Float64}, dt::Float64, dynamics_parameters::Array{Float64})
    γ, η, Γ, n = dynamics_parameters
    ζ = 2.0 * sqrt(η * Γ)
    i1 = ζ * u[1] + randn() / sqrt(dt)
    i2 = ζ * u[2] + randn() / sqrt(dt)
    return [i1, i2] #stochastic current
end

function simulation(
    n::Int, #number of stpes 
    u00::Array{Float64},
    u01::Array{Float64},
    dt::Float64, 
    dynamics_matrix::Function,
    signal_matrix::Function,
    dynamics_parameters0::Array{Float64},
    dynamics_parameters1::Array{Float64},
    #loglike::Function
)
    t = 0
    d = length(u01)
    u1 = Array{Float64}(undef, d, n + 1)
    d = length(u00)
    u0 = Array{Float64}(undef, d, n + 1)
    logl = Array{Float64}(undef, n + 1)
    u0[:, 1] = u00
    u1[:, 1] = u01
    logl[1] = 0
    signaltrace = Array{Float64}(undef, 2, n + 1)
    for i = 1:n
        signal = signal_matrix(u1[:, i], dt, dynamics_parameters1)
        u0[:, i+1] =
            u0[:, i] + dt * dynamics_matrix(u0[:, i], signal, dynamics_parameters0)
        u1[:, i+1] =
            u1[:, i] + dt * dynamics_matrix(u1[:, i], signal, dynamics_parameters1)
        logl[i+1] = (u1[6, i+1] - u0[6, i+1])
        signaltrace[:, i] = signal
        t += dt
    end
    return u0, u1, logl, signaltrace
end





γ1 = 429.
γ0 = 100. #(Hz)
η1 = 1.
η0 = 1.
n1 = 1.
n0 = 1.
Γ1 = 9.
Γ0 = 9.  #(Hz)

σu1 = n1+0.5+Γ1/γ1
σu0 = n0+0.5+Γ0/γ0
#variance of the stationary dynamics.
σ1 = (sqrt(1+16.0*η1*Γ1*σu1/γ1)-1)*γ1/(8.0*η1*Γ1)
σ0 = (sqrt(1+16.0*η0*Γ0*σu0/γ0)-1)*γ0/(8.0*η0*Γ0)

#u0 = [0., 0., σ1, 0., 0., σ0, 0. ,0. ,0.,0.] #
dynamics_parameters0 = [γ0, η0, Γ0, n0]
dynamics_parameters1 = [γ1, η1, Γ1, n1]

u00=[0.0,0.0,σ0, σ0, 0.0,0.0] #6 parameters
u01=[0.0,0.0,σ1,σ1,0.0,0.0]


NN = Int(1e3) #number of element of the ensamble
tf=8#experimental time
tspan = (0.0, tf)
n = Int(1e5)  #number of time steps
dt = (tspan[2]-tspan[1]) /n

vv0 = zeros(NN+1,n+1)
vv1 = zeros(NN+1,n+1)
logl = zeros(NN+1,n+1)
for i=1:N
    println(i)
    u0, u1, logll, signaltrace = simulation(n, u00,u01, dt, dynamics_matrix,signal_matrix, dynamics_parameters0,dynamics_parameters1)
    vv0[i,:] = u0[1,:]
    vv1[i,:] = u1[1,:]
    logl[i,:] = logll
end


npzwrite("data/vv0_1.npy", vv0)
npzwrite("data/vv1_1.npy", vv0)
npzwrite("data/logl_1.npy", logl)
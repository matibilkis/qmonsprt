cd(dirname(@__FILE__))
using LinearAlgebra
using Plots
using BenchmarkTools
using LaTeXStrings
using StatsBase

include("trial.jl")
#plotly()
#gr()
pgfplotsx()
#LinearAlgebra.BLAS.set_num_threads(4)
# Analysis of the  supersmple Gaussian case (see latex file)

############################################

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


u00=[0.0,0.0,σ0, σ0, 0.1,0.0]
u01=[0.0,0.0,σ1,σ1,0.0,0.0]


N = Int(1e3) #number of element of the ensamble
tf=8#experimental time
tspan = (0.0, tf)
n = Int(1e5)  #number of time steps
dt = (tspan[2]-tspan[1]) /n

drive1(t) = [ 0.0 , 0.0] ;
drive0(t) = [  0.0 , 0.0];

vv0 = zeros(NN+1,n+1)
vv1 = zeros(NN+1,n+1)
logl = zeros(NN+1,n+1)
for i=1:NN
    u0, u1, logl, signaltrace = simulation(n, u00,u01, dt, dynamics_matrix,signal_matrix,drive0, drive1, dynamics_parameters0,dynamics_parameters1)
    vv0[i,:] = u0[1,:]
    vv1[i,:] = u1[1,:]
    logl[i,:] = logl[i,:] 
end


x0x0 =zeros(n+1) 
x1x1 = zeros(n+1) 

for i=1:n+1 
    x0x0[i] = cov(vv0[:,i], vv0[:,i])
    x1x1[i] = cov(vv1[:,i], vv1[:,i])
end
x0x0




Δ=Int(1e2)

plot(2*x0x0[1:Δ:end])
#hline!([0.3471])
hline!([0.6889])
plot(2*x1x1[1:Δ:end])
#hline!([0.9165])
hline!([0.31256])



#Stopped simulation.
B= 10. #Upper boundary for the loglik
A=-B  # Lower boundary for the loglike
@time u0_s, u1_s, logl_s, signaltrace_s = stopping_time_simulation(A,B,n, u00,u01, dt,dynamics_matrix, signal_matrix,drive0, drive1, dynamics_parameters0,dynamics_parameters1)
#
Δ=Int(1e2)
t_stop = (length(logl_s)-1)*dt
ts= 0.0:Δ*dt:tf
ts1= 0.0:Δ*dt:t_stop

plot(ts1,u1_s[1,1:Δ:end])
plot!(ts1,u0_s[1,1:Δ:end])
#plot(ts,uu[1,1:Δ:end])
#Plots for the change point.
plot(ts,u1[1,1:Δ:end])
plot!(ts,u0[1,1:Δ:end])
plot!(ts,logl[1:1:end])

length(u1_s)
t_stop/dt
plot(ts1,logl_s[1:Δ:end],label= "")

d= plot(ts, t -> logl_exact(dynamics_parameters0,dynamics_parameters1,t)[4], label = "analytical solution",legend=:topright, line = 2, xaxis = ("time", (0, 3), 0:0.05:0.5))

plot!(ts, t -> logl_exact_new(dynamics_parameters0,dynamics_parameters1,t), label = "analytical solution1",legend=:topright, line = 2, xaxis = ("time", (0, 3), 0:0.05:0.5))

for i=1:10
    u0_s, u1_s, logl_s, signaltrace_s = stopping_time_simulation(A,B,n, u00,u01, dt,dynamics_matrix, signal_matrix,drive0, drive1, dynamics_parameters0,dynamics_parameters1)
    t_stop = (length(logl_s)-1)*dt
    ts1= 0.0:Δ*dt:t_stop
    d= plot!(ts1,logl_s[1:Δ:end], label= "" , xaxis = ("time"))
end
d= yaxis!("loglikelihood Ratio",(-4, 4), -3:1:3)
plot!(ts,t->B ,line=[:dash],label= "")
plot!(ts,t->-B ,line=[:dash],label= "")

savefig("./Plot_lolike_pres.pdf")
#plot(ts,signaltrace[1,1:Δ:end])

#avg, pl = avli_sign(N,n,u00,u01,dt,dynamics_matrix,signal_matrix,drive0,drive1,dynamics_parameters0,dynamics_parameters1)
@time avg, pl = avli(N,n,u00,u01,dt,dynamics_matrix,signal_matrix,drive0,drive1,dynamics_parameters0,dynamics_parameters1)

plot(ts,avg[3,1:Δ:end]/((4.0 * η1 * Γ1 * σu1)^2))

#avg, pl = avli2(N,n,u00,u01,dt,dynamics_matrix,signal_matrix,drive0,drive1,dynamics_parameters0,dynamics_parameters1)

#=
@time pa,pb, avg_n  = avg_stopping_time(A,B,N,n,u00,u01,dt,
                                        dynamics_matrix,signal_matrix,drive0,
                                        drive1,dynamics_parameters0,dynamics_parameters1)
=#


#Plot for the averaged solution (analytical and numerical comparison!)d= plot(ts, t -> logl_exact(dynamics_parameters0,dynamics_parameters1,t)[4], label = "analytical solution", line = 2, xaxis = ("time", (0, 3), 0:0.05:3))
plot!(ts,avg[1,1:Δ:end], label = "avg ($N )", line = 2, xaxis = ("time"))
#title!("TITLE")
d= yaxis!("loglikelihood Ratio",(0, 12), 0:2:12)


savefig("./Plot_avg.pdf")


d= plot(ts, t -> logl_exact(dynamics_parameters0,dynamics_parameters1,t)[4], label = "analytical solution",legend=:topright, line = 2, xaxis = ("time", (0, 50), 0:10:50))
for i=1:5
    u0, u1, logl, signaltrace = simulation(n, u00,u01, dt, dynamics_matrix,signal_matrix,drive0, drive1, dynamics_parameters0,dynamics_parameters1)
    plot!(ts,logl[1:Δ:end], label= "" , xaxis = ("time"))
end
d= yaxis!("loglikelihood Ratio",(-3, 15), -3:2:15)
#d= plot!(ts, t -> logl_exact(dynamics_parameters0,dynamics_parameters1,t)[4], label = "analytical solution", line = 2, xaxis = ("time", (0, 50), 0:10:50))

plot!(ts,t->B ,line=[:dash])
plot!(ts,t->-B ,line=[:dash])

#plot!(size=(750,750))
savefig("./Plot_curves_avg.pdf")
















###################################################
#this is just code that I used to test the probability
σu1 = n1 + 1 / 2 + Γ1 / γ1
σu0 = n0 + 1 / 2 + Γ0 / γ0

σ1 = (sqrt(1 + 16.0 * η1 * Γ1 * σu1 / γ1) - 1) * γ1 / (8.0 * η1 * Γ1)
σ0 = (sqrt(1 + 16.0 * η0 * Γ0 * σu0 / γ0) - 1) * γ0 / (8.0 * η0 * Γ0)

λ = γ0 + 8.0 * η0 * Γ0 * σ0

a = (4 * η1 * Γ1 * (σ1^2)) / γ1
b =
    (4 * η0 * Γ0 * σ0^2) * (
        1 +
        ((16.0 * η1 * Γ1 * σ1) / (γ1 + λ)) +
        (64.0 * (η1 * Γ1 * σ1)^(2) / (γ1 * (γ1 + λ)))
        ) / λ
c =
    8 *
    (σ0 * σ1 * (η0 * Γ0 * η1 * Γ1)^(0.5)) *
    (γ1+ 4.0 * η1 * Γ1 * σ1 ) / ((γ1 + λ)*γ1)

μ = 4*(η1 * Γ1 * a + η0 * Γ0 * b - 2* sqrt(η1 * Γ1 * η0 * Γ0) * c)
σ=  sqrt(2*μ)

P(t,μ,σ,b) = b*exp(-((b-μ*t)^2)/(2*t*σ^2))/(t^(3/2)*sqrt(2*π)*σ)
P1(t,μ,σ,b) = (b-2*μ*t)*exp(-((b-μ*t)^2)/(2*t*σ^2))/(t^(3/2)*sqrt(2*π)*σ)


plot(ts,t->P(t,μ,σ,B)-P(t,μ,σ,-B))
plot!(ts,t->P(t,μ,σ,B))
plot!(ts1,t->-P(t,μ,σ,-B))

ts1 =0:0.05:100.0000001


ts

function Prob(t,μ,σ,b,a,n)
    tot=0
    P(t,μ,σ,b) = b*exp(-((b-μ*t)^2)/(2*t*σ^2))/(t^(3/2)*sqrt(2*π)*σ)
    P1(t,μ,σ,b) = abs((b-2*μ*t))*exp(-((b-μ*t)^2)/(2*t*σ^2))/(t^(3/2)*sqrt(2*π)*σ)
    for i= 0:n
        tot+=P(t,μ,σ,b-2*n*(b-a))-P1(t,μ,σ,b-(2*n+1)*(b-a))
           -P(t,μ,σ,a+2*n*(b-a))-P1(t,μ,σ,a+(2*n+1)*(b-a))
    end
    return a
end

Prob(t,μ,σ,b,100)
P(t,μ,σ,b) = b*exp(-((b-μ*t)^2)/(2*t*σ^2))/(t^(3/2)*sqrt(2*π)*σ)

ts =0:0.0025:1

plot(ts,t->P(t,μ,σ,B),label="")
d= yaxis!(L"P(\tau=t)")
d= xaxis!(L"t")
ts1 =0:0.05:200.0000001
savefig("./Plot_approx_prob_cu*rve_pres.pdf")



using StatsBase, Plots

h = fit(Histogram, rand(10000000), nbins=100)
plot(h)

sample_n=Int(1e3)
times = zeros(sample_n)
for i=1:sample_n
    u0_s, u1_s, logl_s, signaltrace_s = stopping_time_simulation(-100.,B,n, u00,u01, dt,dynamics_matrix, signal_matrix,drive0, drive1, dynamics_parameters0,dynamics_parameters1)
    t_stop = (length(logl_s)-1)*dt
    times[i]=t_stop
    #push!(times,t_stop)
end

h = fit(Histogram, times, nbins=200)
hn=normalize(h, mode=:pdf)
plot(hn)


μ = 4*(η1 * Γ1 * a + η0 * Γ0 * b - 2* sqrt(η1 * Γ1 * η0 * Γ0) * c)
σ=  sqrt(2*μ)
σ^2

P(t,μ,σ,b) = b*exp(-((b-μ*t)^2)/(2*t*σ^2))/(t^(3/2)*sqrt(2*π)*σ)
ts =0:0.01:1.5
plot!(ts,t->P(t,μ,σ,B),label="")

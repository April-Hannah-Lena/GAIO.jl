#ENV["JULIA_DEBUG"] = "all"
using GAIO

# -----------------------------------------

N = 3
const σ, ρ, β = 10.0, 28.0, 0.4
function f(x)
    dx = (
           σ * x[2] -    σ * x[1],
           ρ * x[1] - x[1] * x[3] - x[2],
        x[1] * x[2] -    β * x[3]
    )
    return dx
end
F(x) = rk4_flow_map(f, x)
center, radius = (0,0,25), (30,30,30)
P = BoxPartition(Box(center, radius), (128,128,128))
x = (sqrt(β*(ρ-1)), sqrt(β*(ρ-1)), ρ-1)# .+ tuple(15.0 .* rand(3)...)

# -----------------------------------------

println("\nStandard Version")
@show G = BoxMap(F, P)
Y = G(P[x])
@show W = unstable_set!(G, P[x])
for _ in 1:10
    try
        @time unstable_set!(G, P[x])
    catch
        print("errorred")
    end
end

# -----------------------------------------

println("\ncpu version")
include("redef_map_boxes.jl")
@show G = BoxMap(F, P, :cpu)
Y = G(P[x])
@show W = unstable_set!(G, P[x])
for _ in 1:10
    try
        @time unstable_set!(G, P[x])
    catch
        print("errorred")
    end
end

#plot(W)
using GAIO
using SparseArrays, LinearAlgebra, KrylovKit
using ProgressMeter, Base.Threads, Serialization
using DSP

using GLMakie
const Box = GAIO.Box

# --------------------------------

f( x::Real; α=2, β=-1-exp(-α) ) = exp(-α*x^2) + β
f( (x,); kwargs... ) = ( f(x; kwargs...) ,)

fig, ax, ms = plot(-1:0.001:0, f, color=:blue);
ms = plot!(ax, -1:0.001:0, identity, color=:green)
fig

domain = Box([-0.5], [0.5])
P = BoxPartition(domain, (1024,)) 
S = cover(P, :)
F = BoxMap(:interval, f, domain)

# ---------------------------------

function f( x::Real; s=4-1/8 )
    if 0 ≤ x < 1/4
        2x
    elseif 1/4 ≤ x < 1/2
        s * (x - 1/4)
    elseif 1/2 ≤ x < 3/4
        s * (x - 3/4) + 1
    elseif 3/4 ≤ x ≤ 1
        2 * (x - 1) + 1
    else
        @error "how did you get here" x
    end
end

f( (x,); s... ) = ( f(x; s...) ,)

domain = Box([0.5], [0.5])
P = BoxPartition(domain, (1024,))
S = cover(P, :)
F = BoxMap(:grid, f, domain)

# ---------------------------------

F♯ = TransferOperator(F, S, S)

M = similar(F♯.mat', ComplexF64)
M .= F♯.mat'

n = size(M, 1)
x0 = rand(ComplexF64, n) .* exp.(2pi*im .* rand(Float64, n))

function res(z; kwargs...)
    vals, rvecs, lvecs, info = svdsolve(M - z*I, x0, 1, :SR; kwargs...)
    return minimum(vals)
end

xs = -1.1:0.005:1.1
R = ones(Float64, length(xs), length(xs))

prog = Progress(length(R))
@threads for cart in CartesianIndices(R)
    i, j = Tuple(cart)
    z = xs[i] + xs[j] * im

    R[cart] = res(z)
    next!(prog)
end

#=R = @showprogress broadcast(xs', xs) do x, y
    res(x+y*im)
end=#

serialize("resolvents.ser", R)
R = deserialize("resolvents.ser")

# remove basically-0 revolvents
R̄ = R
#R̄[R̄ .> 3e-1] .= 3e-1

# smoothing
#A = 0.25 * [1 1;
#            1 1]

#R̄ = conv(R̄, A)[1:end-1, 1:end-1]

λ, ev, nconv = eigs(F♯, nev=100, which=:LM)

resolvents = @showprogress map(res, λ)

serialize("eigresolvents.ser", resolvents)
resolvents = deserialize("eigresolvents.ser")

perm = sortperm(resolvents)
λ .= λ[perm]
resolvents .= resolvents[perm]

ticks = -8:1:-1
labs = ["10^$x" for x in ticks]

fig, ax, ms = contour(xs, xs, log10.(R̄)) 
ms = scatter!(ax, real.(λ), imag.(λ), color=:blue)

p2 = heatmap(xs, xs, log10.(R̄))

fig = Figure();
ax = Axis3(fig[1,1], aspect=(1,1,1))
ms = surface!(ax, xs, xs, log10.(R̄), colormap=(:viridis, 0.5))
ms = scatter!(ax, real.(λ), imag.(λ), log10.(resolvents), color=:blue)
fig

ys = -0.05:0.01:0.05
Ri = ones(length(ys), length(ys), 20)
for i in axes(Ri, 3)
    indices = CartesianIndices(Ri[:, :, i])
    prog = Progress( length(indices) )
    @threads for cart in indices
        j, k = Tuple(cart)
        z = ys[j] + ys[k] * im
        z += λ[i]

        Ri[j, k, i] = res(z)
        next!(prog)
    end

    global ms = surface!(ax, 
        real(λ[i]) .+ ys, 
        imag(λ[i]) .+ ys,
        log10.(Ri[:, :, i]),
        colormap=(:viridis, 0.5)
    )
end
fig
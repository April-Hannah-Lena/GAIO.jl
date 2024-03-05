using GAIO
using SparseArrays, LinearAlgebra, KrylovKit
using ProgressMeter, Base.Threads, Serialization
using DSP

using GLMakie
const Box = GAIO.Box

# --------------------------------

_f( x::Real; α=2, β=-1-exp(-α) ) = exp(-α*x^2) + β
f( x::Real; kwargs... ) = 2 * _f( (x-1)/2; kwargs... ) + 1
f( (x,); kwargs... ) = ( f(x; kwargs...) ,)

fig, ax, ms = plot(-1:0.002:1, f, color=:blue);
ms = plot!(ax, -1:0.002:1, identity, color=:green)
fig

domain = Box([0.], [1.])
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





# ResDMD

using LinearAlgebra, LegendrePolynomials, FastGaussQuadrature, ProgressMeter, PROPACK, GAIO, Arpack
using Plots

function _f( x::Real; s=4-1/8 )
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

f( x::Real; s... ) = 2 * _f( (x+1)/2; s... ) - 1
f( (x,); s... ) = ( f(x; s...) ,)

domain = Box([0.], [1.])
P = BoxPartition(domain, (1024,))
S = cover(P, :)

F = BoxMap(:grid, f, domain)
F♯ = TransferOperator(F, S, S)

#anim = @animate for n in 60:100:1500
n = 1160
nodes, weights = gausslegendre(n)
#nodes = [range(-1, 1, length=100);]
#weights = fill!(similar(nodes), 1/100)

Ψ₀ = reduce(hcat, collectPl.(nodes, lmax = 59))'
Ψ₁ = reduce(hcat, collectPl.(f.(nodes), lmax = 59))'

W = Diagonal(weights)

G = Ψ₀' * W * Ψ₀
G ≈ Diagonal(G) ? (G = Diagonal(G)) : @error "not orthogonal"
B = Diagonal(1 ./ sqrt.(G.diag))
Ψ₀ *= B
Ψ₁ *= B

G = Ψ₀' * W * Ψ₀
G = G ≈ Diagonal(G) ? Diagonal(G) : G
DG, VG = eigen(G)
DG = Diagonal(DG)
SQ = VG * DG * VG'

A = Ψ₀' * W * Ψ₁
A[A .< eps(eltype(A))] .= 0

L = Ψ₁' * W * Ψ₁
L[L .< eps(eltype(L))] .= 0

dom = [-1.1:0.02:1.1;]

res = @showprogress broadcast(dom, dom') do zx, zy
    z = zx + zy * im
    #λ, _ = tsvdvals_irl(L - z * A - conj(z) * A' + abs(z)^2 * G, k=1)
    λ = eigvals(SQ * (L - z * A - conj(z) * A' + abs(z)^2 * G) * SQ)
    real( λ[ argmin(abs.(λ)) ] )
end

contourf(
    dom, dom, log10.(res'), 
    levels=50, size=(900,700), color=:haline,
    title="$n quadrature points, 60 dictionary polynomials"
)

λ, ev, nconv = eigs(F♯, nev=100, which=:LM)
scatter!(λ, color=:pink, markersize=5)

λ, ev, nconv = eigs(A, G, nev=100, which=:LM)
scatter!(λ, color=:red, marker=:xcross, markersize=6)

#end
gif(anim, fps=0.1)
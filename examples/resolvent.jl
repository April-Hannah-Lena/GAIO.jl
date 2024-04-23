using GAIO
using SparseArrays, LinearAlgebra, KrylovKit
using ProgressMeter, Base.Threads, Serialization
using DSP

using GLMakie
const Box = GAIO.Box

# --------------------------------

f( x::Real; α=2, β=-1-exp(-α) ) = exp(-α*x^2) + β
f( (x,); kwargs... ) = ( f(x; kwargs...) ,)

Df( x::Real; α=2, β=-1-exp(-α) ) = -2α*x*exp(-α*x^2)

fig, ax, ms = plot(-1:0.002:0, f, color=:blue);
ms = plot!(ax, -1:0.002:0, identity, color=:green)
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





# ResDMD

using LinearAlgebra, LegendrePolynomials, FastGaussQuadrature, ChebyshevApprox, ProgressMeter, PROPACK, GAIO, Arpack
using Plots

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

#f( x::Real; s... ) = 2 * _f( (x+1)/2; s... ) - 1
f( (x,); s... ) = ( f(x; s...) ,)

domain = Box([0.5], [0.5])
P = BoxPartition(domain, (1024,))
S = cover(P, :)

F = BoxMap(:grid, f, domain)
F♯ = TransferOperator(F, S, S)

#anim = @animate for n in 60:100:1500
n = 1000
R = 500

#=nodes = [range(-1, 1, length=100);]
weights = fill!(similar(nodes), 1/100)=#

nodes, weights = gausslegendre(n)
data = 2 .* f.((nodes .+ 1) ./ 2) .- 1
ΨX = reduce(hcat, collectPl.(nodes; norm=Val(:normalized), lmax = R-1))'
ΨY = reduce(hcat, collectPl.(data; norm=Val(:normalized), lmax = R-1))'

#=nodes = ChebyshevApprox.nodes(n, :chebyshev_nodes).points
weights = π/n * ones(n)

φ0(x) = 1
function φ1(x)
    if -1 ≤ x ≤ 0
        1 + cospi(2x+1)
    else
        0
    end
end

φ = [φ0, φ1]
for k in 1:5
    for r in 0:2^k - 1
        push!(φ, (y -> 2^k * y) ∘ φ1 ∘ (x -> 2^k * x + r))
    end
end

#ΨX = chebyshev_polynomial(39, nodes) .* sqrt.(1 .- nodes .^ 2)
#ΨX = chebyshev_polynomial(39, 2 .* f.((nodes .- 1) ./ 2) .+ 1) .* sqrt.(1 .- nodes .^ 2)

ΨX = Float64[φi(x) for x in nodes, φi in φ]
ΨY = Float64[φi(x) for x in 2 .* f.((nodes .- 1) ./ 2) .+ 1, φi in φ] =#

W = Diagonal(weights)

G = ΨX' * W * ΨX
#G[G .< 10*eps()] .= 0
if G ≈ Diagonal(G)
    G = Diagonal(G)
else
    G = Symmetric((G + G') / 2)
end

#B = Diagonal(1 ./ sqrt.(G.diag))
#Ψ₀ *= B
#Ψ₁ *= B

#G = Ψ₀' * W * Ψ₀
#G = G ≈ Diagonal(G) ? Diagonal(G) : G
DG, VG = eigen(G)
DG = Diagonal(DG)
#DG[DG .< 10*eps()] .= 0
DG[DG .> 0] .= 1 ./ sqrt.( DG[DG .> 0] )

SQ = VG * DG * VG'
#SQ[SQ .< 10*eps()] .= 0
if SQ ≈ Diagonal(SQ)
    SQ = Diagonal(SQ)
else
    SQ = Symmetric((SQ + SQ') / 2)
end

A = ΨX' * W * ΨY
#A[A .< 10*eps()] .= 0

L = ΨY' * W * ΨY
#L[L .< 10*eps()] .= 0
if L ≈ Diagonal(L)
    L = Diagonal(L)
else
    L = Symmetric((L + L') / 2)
end

function residual(z, SQ=SQ, G=G, A=A, L=L)
    λ = eigvals(SQ * (L - z * A' - conj(z) * A + abs(z)^2 * G) * SQ)
    λ0 = λ[ argmin(abs.(λ)) ]
    real(λ0) < 0 && @warn "erroneous negative residual calculated" z calculated_residual=λ0
    sqrt(max(real(λ0), 0))
end

λ, ev, nconv = eigs(F♯', nev=30)
λ_res = residual.(λ)
λ_dmd, ev_dmd, nconv = eigs(A; nev=30)

ys = [-1.025:0.025:1.025;]
xs = [-1.025:0.025:1.5;]

res = @showprogress broadcast(xs', ys) do x, y
    z = x + y * im
    if abs(z) > 1.1 && x < 1
        NaN
    else
        residual(z)
    end
end

xs2 = 0.65:0.005:0.75
ys2 = -0.1:0.01:0.1
res2 = @showprogress broadcast(xs2', ys2) do x, y
    residual(x + y * im)
end

levels = [0:0.05:0.35;]#[0.001, 0.05, 0.13, 0.16, 0.2, 0.3]

contour(
    xs, ys, #=log10.(=#res#=)=#, 
    levels=levels,#10 .^ (-2:0.01:0),#[0.001,0.01,0.1,0.3], 
    clims=(0., 0.35),
    size=(900,700), color=:rainbow, aspectratio=1., clabels=true, 
    title="$n quadrature points, $R Legendre polynomials"
)

contour!(xs2, ys2, res2, levels=[levels; 0.13; 0.132], color=:lightrainbow, clabels=true)

#scatter!(λ[λ_res .< 0.2], color=:blue, marker=:xcross, markersize=6, lab=false)
scatter!(λ, color=:red, alpha=0.8, marker=:xcross, label="Ulam eigs")
scatter!(λ_dmd, color=:blue, alpha=0.8, marker=:cross, label="EDMD eigs")
plot!(cos.(0:0.01:2π), sin.(0:0.01:2π), lab="|z|=1", style=:dash, color=:black, alpha=0.4)

savefig("./paper/pseudospec_fourlegs.png")

μ = real ∘ ev[2]
v = real.(ev_dmd[:, 2])

g = reduce(hcat, collectPl.(-1:0.0002:1; norm=Val(:normalized), lmax = R-1))'
g .*= v'
g = g * ones(R)

plot(μ)
savefig("./paper/ulam_eig.png")
plot(0:0.0001:1, g)
savefig("./paper/dmd_eig.png")


using DataFrames, CSV, DelimitedFiles

matlab_A = Matrix(CSV.read("../Residual-Dynamic-Mode-Decomposition/A.csv", DataFrame, header=false))
matlab_L = Matrix(CSV.read("../Residual-Dynamic-Mode-Decomposition/L.csv", DataFrame, header=false))

errs_A = abs.(A - matlab_A)
errs_L = abs.(L - matlab_L)

writedlm("./examples/A.csv", A, ',')
writedlm("./examples/L.csv", L, ',')

matlab_res = Matrix(CSV.read("../Residual-Dynamic-Mode-Decomposition/res.csv", DataFrame, header=false))

using Polynomials, SpecialFunctions

𝟙 = Polynomial([1.])
𝕩 = Polynomial([0., 1.])
legendres = [𝟙, 𝕩]

for degree in 2:39
    l = degree - 1
    push!(
        legendres,
        ( (2*l+1) * 𝕩 * legendres[end] - l * legendres[end-1] ) / (l + 1)
    )
end

# ∫_0^{√kα} x^l erf(x) dx
function H(k, l, α=2.)
    k == 0 && return 0
    l == 0 && return sqrt(k*α)*erf(sqrt(k*α)) + (exp(-k*α) - 1) / sqrt(π)

    out  = (k*α)^((l+1)/2) * erf(sqrt(k*α))
    out += (k*α)^(l/2) * exp(-k*α) / sqrt(π)
    out -= l * SE(k, l-1) / 2
    out /= l + 1
    
    return out
end

# ∫_0^{√kα} x^l 2/√π e^{-x^2} dx
function SE(k, l, α=2.)
    k == 0 && return 0
    l == 0 && return erf(sqrt(k*α))

    out  = (k*α)^(l/2) * erf(sqrt(k*α))
    out -= l * H(k, l-1)

    return out
end

function σ(n, k, m, α=2., β=-1-exp(-α))
    binomial(n, k) * β^(n-k) * (k*α)^(-(m+1)/2)
end

# ∫₋₁⁰  F(x)^n ⋅ x^m dx
function Kxⁿ_xᵐ(n, m, α=2., β=-1-exp(-α))
    out = β^n / (m+1)
    if n > 0
        out += sum(σ(n, k, m) * SE(k, m) for k in 1:n) * sqrt(π) / 2
    end
    out *= (-1)^m
    return out
end

𝕙 = Polynomial([1., 2.])

function Base.:(∘)(g::Polynomial, h::Polynomial)
    sum(coeff * h^deg for (deg,coeff) in pairs(g))
end

# ∫₋₁⁰  2 ⋅ (Pₙ(2⋅ + 1) ∘ F)(x) ⋅ Pₘ(2x + 1) dx
function KPₙ_Pₘ(n, m)
    Pn = legendres[n+1] ∘ 𝕙
    Pm = legendres[m+1] ∘ 𝕙
    #@info "polynomials" Pn Pm

    sum(
        coeff_n * coeff_m * Kxⁿ_xᵐ(deg_n, deg_m)
        for (deg_n,coeff_n) in pairs(Pn), (deg_m,coeff_m) in pairs(Pm)
    )
end

true_ΨX_W_ΨY = [KPₙ_Pₘ(m,n) for m in 0:39, n in 0:39]




_ϵ(::Nothing) = 0
_ϵ(x) = x

function res_ulam(z, μ; montecarlo_points=2rand(1000).-1)
    S = F♯.domain
    P = S.partition

    for Ai in S, Aj in S
        ν = BoxFun(P, Dict(point_to_key(P, Ai.center)=>1.))
        @info "dbg" [values(ν);]

        F⁻¹Aj = BoxSet(F♯'ν)
        @info "dbg" first(F⁻¹Aj)

        out = volume(Ai ∩ Aj) - 2*real(z)*sum(volume(Ai ∩ B) for B in F⁻¹Aj if !isnothing(Ai ∩ B))

        FAi_n_FAj = BoxSet(F♯*ν)
        ν = BoxFun(P, Dict(point_to_key(P, Aj.center)=>1.))
        FAi_n_FAj = FAi_n_FAj ∩ BoxSet(F♯*ν)

        for B in FAi_n_FAj
            out += μ(B) / volume(B) * sum(x->1/Df(x), montecarlo_points)
        end
    end

    return sqrt(out)
end

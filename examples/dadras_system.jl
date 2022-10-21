using LinearAlgebra, StaticArrays, MuladdMacro
using GLMakie: plot
using GAIO

function Sdiagm(factor::T, size) where T
    SMatrix{size,size,T}(
        ntuple(
            i -> (i - 1) % (size + 1) ? factor : zero(T),
            size ^ 2
        )
    )
end

# (scaled) Dadras system
# we use a coordinate transformation x̃ = μ(x)
# with μ(x) = x * η(x), η(x) = 1 / (sqrt ∘ norm)(x)
const ee = 2 * eps() ^ (1/3)
const a, b, c = 8.0, 40.0, 14.9
function v(x::SVector{4,T}) where {T}
    #η = 1 / max((sqrt ∘ norm)(x), ee)   # to ensure we dont get Inf
    η = 1 / (sqrt ∘ norm)(x)
    ∇η = -x .* (η ^ 3) ./ 2
    #Dμ = Sdiagm(η) .+ ∇η
    fx = let α = x[1], β = x[2], δ = x[3], γ = x[4]
        SVector{4,T}(a*α-β*δ+γ, α*δ-b*β, α*β-c*δ+α*γ, -β)
    end
    #fx̃ = Dμ * fx
    #s = ∇η'fx
    #s = x .* s
    fx̃ = η .* fx .+ x .* (∇η'fx)
end
f(x) = rk4_flow_map(v, x, 0.01, 1)

# The system is extremely expansive, so resolving the entire box image is 
# difficult. Hence we try with an adaptive test point sampling approach 
# that attempts to handle errors due to the map diverging. 

const montecarlo_points = [ SVector{4,Float64}(2f0*rand(4).-1f0 ...) for _ = 1:4000 ]

function domain_points(center::SVector{N,T}, radius::SVector{N,T}) where {N,T}
    try
        L, y, n, h = zeros(T,N,N), MVector{N,T}(zeros(T,N)), MVector{N,Int}(undef), MVector{N,T}(undef)
        fc = f(center)
        for dim in 1:N
            y[dim] = radius[dim]
            fr = f(center .+ y)
            L[:, dim] .= abs.(fr .- fc) ./ radius[dim]
            y[dim] = zero(T)
        end
        all(isfinite, L) || @error "The dynamical system diverges within the box." box=Box{N,T}(center, radius)
        _, σ, Vt = svd(L)
        n .= ceil.(Int, σ)
        h .= 2.0 ./ (n .- 1)
        points = Iterators.map(CartesianIndices(ntuple(k -> n[k], Val(N)))) do i
            p = [n[k] == 1 ? zero(T) : (i[k] - 1) * h[k] - 1 for k in 1:N]
            p .= Vt'p
            @muladd p .= center .+ radius .* p
            sp = SVector{N,T}(p)
        end
        return points
    catch ex
        @warn "$ex was thrown when calculating adaptive points within the box." box=Box{N,T}(center, radius)
        return (@muladd(center .+ radius .* point) for point in montecarlo_points)
    end
end

image_points(center, radius) = vertices(center, radius)

domain = Box((0,0,0,0), (250,150,200,25))
P = BoxPartition(domain, (32,32,32,32))

F = SampledBoxMap(
    f, 
    domain,
    domain_points,
    image_points,
    nothing
)

F = BoxMap(f, P, no_of_points=40)

#x = zeros(4)        # equilibrium
equillibrium = P[Box((0,0,0,0), (0.1,0.1,0.1,0.1))]
@profview W = unstable_set!(F, equillibrium)

P = BoxPartition(domain)
W = relative_attractor(F, P[:], steps=20)


plot(W)

#T = TransferOperator(W)
#(λ, ev) = eigs(T)

#plot(log∘abs∘ev[1])

#using GAIO
using SparseArrays, StaticArrays, LinearAlgebra, Arpack
using LegendrePolynomials, FastGaussQuadrature, DoubleFloats
using ProgressMeter, Plots
using ThreadsX
using Base.Threads: nthreads, @threads

macro exitsafe(expr)
    return quote
        try
            $(esc(expr))
        catch err
            err isa InterruptException && rethrow()
            println("-"^30)
            showerror(stdout, err)
            println("-"^30)
            NaN
        end
    end
end

const μ = 0.8 * exp(im*pi/8)
const ρ = 0. + 0im
const params = (μ, ρ)

τ(z, p = params) = prod( @. (z - p)/(1 - conj(p) * z) )
#τ(z, μ=, ρ=0.) = (z - μ)/(1 - μ' * z) * (z - ρ)/(1 - ρ' * z)

# for μ ∈ int(𝔻),  ρ = 0,  params = (μ, ρ)
const λ_true = sort!( 
    vec( [(-μ) (-μ)'] .^ (0:50) ), 
    by=abs, rev=true 
)

#=
begin

    function gauss(x::Real; α=2, β=-1-exp(-α))
        exp(-α*x^2) + β
    end

    gauss((x,); α=2, β=-1-exp(-α)) = (gauss(x; α=α, β=β),)

    jac_gauss(x::Real; α=2) = -2*α*x*exp(-α*x^2)
    jac_gauss((x,); α=2) = (jac_gauss(x; α=α),)

    function inv_gauss(y::Real; α=2, β=-1-exp(-α))
        exp(-α) ≤ y - β ≤ 1  ||  throw(DomainError(y))
        -sqrt( -log(y-β)/α )>
    end

    inv_gauss((x,); α=2, β=-1-exp(-α)) = (inv_gauss(x; α=α, β=β),)

    function jac_inv_gauss(y::Real; α=2, β=-1-exp(-α))
        exp(-α) ≤ y - β ≤ 1  ||  return 0
        -1 / ( 2α * (y - β) * inv_gauss(y; α=α, β=β) )
    end

    jac_inv_gauss((x,); α=2, β=-1-exp(-α)) = SMatrix{1,1}(jac_inv_gauss(x; α=α, β=β))

    doubling(x::Real) = (2x % 1) + 0.05*sinpi(4x)
    doubling((x,)) = (doubling(x),)

    function mob(θ::Real, α=0.8*exp(2π*im/9))
        z = exp(2*π*im*θ)
        z *= (z - α) / (1 - α' * z)
        (angle(z) / π + 1) / 2
    end

    mob((x,)) = (mob(x),)

    function linear_gauss(_x::Real)
        x = _x + 1
        p = if x < 1/4
            -4x
        elseif x < 3/4
            2(x - 3/4)
        else
            -4(x - 3/4)
        end
        (x + p/10) - 1
    end

    linear_gauss((x,)) = (linear_gauss(x),)

    function inv_linear_gauss(_x::Real)
        x = _x + 1
        p = if x < 1/4
            5x/3
        elseif x < 3/4
            (x + 3/20) * 5/6
        else
            (x - 3/10) * 5/3
        end
        p - 1
    end

    function jac_inv_linear_gauss(_x::Real)
        -3/4 ≤ _x < -1/4 ? 5/6 : 5/3
    end

    function jac_linear_gauss(_x::Real)
        x = _x + 1
        p = ( 1/4 ≤ x < 3/4 )  ?  2 : -4
        1 + p/10
    end>c_linear_gauss(x),)

    ∘ⁿ(f, n) = f ∘ ( n == 1 ? identity : ∘ⁿ(f, n-1) )

    plot(-1:0.001:0, [identity, gauss, linear_gauss], label=["identity" "Gauss Map" "'Crude' linear approx."])

    savefig("../talk_GAIO.jl/maps.png")

end


cen, rad = (0.5,), (0.5,)
dom = Box(cen, rad)
P = BoxPartition(dom, (128,))
S = cover(P, :)
=#

const xs = -1.05:0.01:1.2
const ys = -1.05:0.01:1.1

#=
function res_ulam(
        f = gauss, 
        D_invf = jac_inv_gauss, 
        S = S, 
        xs = xs, 
        ys = ys
    )

    F = BoxMap(f, S.partition.domain)
    F♯ = TransferOperator(F, S, S)
    S = F♯.domain

    λ, ev, nconv = eigs(F♯', nev=100)

    # easy because all same, otherwise use vector
    vols = volume(first(S)) #[volume(Ai) for Ai in S]
    G = UniformScaling(vols) #Diagonal(vols)

    A = sparse(F♯)' * G

    # easy because const on boxes, otherwise we need some quadrature rule
    L = Diagonal([ det(D_invf(c)) for (c,r) in S ]) * G

    function residual(z, G=G, A=A, L=L)
        ξ#=, _=# = eigvals( 
            Matrix(inv(G) * (L - z * A' - z' * A + abs(z)^2 * G)),
            #which=:SM, nev=8, ritzvec=false, ncv=40, tol=0.0
        )
        ξ0 = ξ[ argmin(abs.(ξ)) ]
        real(ξ0) < 0 && @warn "erroneous negative residual calculated" z calculated_residual=ξ0
        #sqrt( max( real(ξ0), 0 ) )
        sqrt( abs(ξ0) )
    end

    res = @showprogress broadcast(xs', ys) do x, y
        z = x + y * im
        if (abs(z) > 1.2 && x < 1) || abs(z) > 1.25
            NaN
        else
            @exitsafe residual(z)
        end
    end

    return res, λ
end

begin

nodes, weights = gausslegendre(1024)
nodes = Double64.(nodes)
weights = Double64.(weights)
nodes = [-1:0.001:1;]
weights = 2 .* ones(length(nodes)) ./ length(nodes)

function res_dmd(
        f = gauss,
        nodes = nodes,
        weights = weights,
        basis = leg(Val(128)),#x -> [1/sqrt(2); cospi.(x .* (1:14)); sinpi.(x .* (1:14))],
        xs = xs, 
        ys = ys
    )
    
    data = 2 .* f.( (nodes .- 1) ./ 2 ) .+ 1

    nodes = [-1:0.01:1;]
    weights = ones(length(nodes)) ./ length(nodes)

end
=#

const M = 2_000_000
const R = 11/10 - 1/32
const r = 1/R
const N = 30

function main()

dθ = 1/M
θs = dθ:dθ:1

basis( z, n, d = sqrt(r^(2n) + R^(2n)) ) = z^n / d
basis(n) = z -> basis(z, n)

G = zeros(ComplexF64, 2N+1, 2N+1)
A = zeros(ComplexF64, 2N+1, 2N+1)
L = zeros(ComplexF64, 2N+1, 2N+1)

prog = Progress((2N+1)^2, desc="Computing matrices...")
@threads for cartesian in CartesianIndices(G)
    i, j = cartesian.I
    ψi = basis( (-N:N)[i] )
    ψj = basis( (-N:N)[j] )
    for ξ in (r, R)
        @inbounds @fastmath @simd ivdep for k in 1:M
            θ = θs[k]
            xk = ξ * exp(2π*im*θ)

            G[i,j] +=   ψi(xk)'      *    ψj(xk)
            A[i,j] +=   ψi(xk)'      *  (ψj ∘ τ)(xk)
            L[i,j] += (ψi ∘ τ)(xk)'  *  (ψj ∘ τ)(xk)
        end
    end
    next!(prog)
end

G ./= M
A ./= M
L ./= M

G .+= G';  G ./= 2
L .+= L';  L ./= 2


function residual(z, G=G, A=A, L=L)
    λ = eigvals( inv(G) * (L - z * A' - z' * A + abs(z)^2 * G) )
    λ0 = λ[ argmin(abs.(λ)) ]
    real(λ0) < 0 && @warn "erroneous negative residual calculated" z calculated_residual=λ0 maxlog=10
    sqrt(max(real(λ0), 0))
end

res = Matrix{Float64}( undef, (length(ys),length(xs)) )
prog = Progress(length(res)^2, desc="Computing residuals...")
@threads for cartesian in CartesianIndices(res)
    i, j = cartesian.I
    y = ys[i];  x = xs[j]
    z = x + y*im
    res[i,j] = if abs(z) > 1.2
        NaN
    else
        @exitsafe residual(z)
    end
    next!(prog)
end

λ = eigvals(A)


#r_ulam, λ_ulam = res_ulam()
r_dmd = res
λ_dmd = λ

levels = sort!([#= 0.00001; 0.0001; 0.001; 0.005; 0.01; 0.02; 0.05;  =#0:0.1:1; 0.58])

#=p1 = contour(
    xs, ys, r_ulam, 
    levels=levels, aspectratio=1., colormap=:rainbow, clabels=true
)
scatter!(λ_ulam, label="Ulam eigs", marker=:xcross)
scatter!(λ_dmd, label="EDMD eigs", marker=:cross)
plot!(cospi.(0:0.01:2), sinpi.(0:0.01:2), style=:dash, label="|z| = 1")=#

p2 = contour(
    xs, ys, r_dmd, 
    levels=levels, aspectratio=1., colormap=:rainbow, clabels=true,
    size=(1200,900)
)
scatter!(λ_ulam[1:10], label="Ulam eigs", marker=:xcross)
scatter!(λ_dmd, label="EDMD eigs", marker=:cross)
scatter!(ComplexF64[((0.8*exp(2π*im/9)) .^ (1:10)); ((0.8*exp(-2π*im/9)) .^ (1:10))], marker=:xcross, label="True eigs")
plot!(cospi.(0:0.01:2), sinpi.(0:0.01:2), style=:dash, label="|z| = 1")

plot(p1, p2, size=(1200,600))

savefig("../talk_GAIO.jl/pseudospectrum_comparison_gauss.png")


anim = @animate for n_basis in [32, 64, 128, 256]
    n_points = 4*n_basis
    nodes, weights = gausslegendre(n_points)
    r_dmd, λ_dmd = res_dmd(mob, nodes, weights, leg(Val(n_basis)))
    p3 = contour(
        xs, ys, r_dmd, 
        levels=levels, aspectratio=1., colormap=:rainbow, clabels=true,
        title="M = $M, N = $N, r = $(round(r, digits=3)), R = $(round(R, digits=3))"
    )
    #scatter!(λ_ulam, label="Ulam eigs", marker=:xcross)
    scatter!(λ_dmd, label="EDMD eigs", marker=:cross)
    plot!(exp.( im .* (-π:0.01:π) ), label="|z| = 1", style=:dash)
    scatter!(λ_true, label="true eigs", marker=:xcross)
#end

savefig(p2, "./pseudospectrum.png")

return
end # function

main()

using GAIO
using SparseArrays, StaticArrays, LinearAlgebra, Arpack
using LegendrePolynomials, FastGaussQuadrature
using ProgressMeter, Plots

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

function gauss(x::Real; α=2, β=-1-exp(-α))
    exp(-α*x^2) + β
end

gauss((x,)) = (gauss(x),)

jac_gauss(x::Real; α=2) = -2*α*x*exp(-α*x^2)
jac_gauss((x,); α=2) = jac_gauss(x; α=α)

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

function jac_linear_gauss(_x::Real)
    x = _x + 1
    p = ( 1/4 ≤ x < 3/4 )  ?  2 : -4
    1 + p/10
end

jac_linear_gauss((x,)) = jac_linear_gauss(x)

∘ⁿ(f, n) = f ∘ ( n == 1 ? identity : ∘ⁿ(f, n-1) )

plot(-1:0.001:0, [identity, gauss, linear_gauss], label=["identity" "Gauss Map" "'Crude' linear approx."])

savefig("../talk_GAIO.jl/maps.png")

cen, rad = (-0.5,), (0.5,)
dom = Box(cen, rad)
P = BoxPartition(dom, (1024,))
S = cover(P, :)

xs = -1.05:0.05:1.2
ys = -1.05:0.05:1.1

function res_ulam(
        f = linear_gauss, 
        Df = jac_linear_gauss, 
        S = S, 
        xs = xs, 
        ys = ys
    )

    F = BoxMap(f, S.partition.domain)
    F♯ = TransferOperator(F, S, S)
    S = F♯.domain

    λ, ev, nconv = eigs(F♯, nev=100)

    # easy because all same, otherwise use vector
    vols = volume(first(S)) #[volume(Ai) for Ai in S]
    G = UniformScaling(vols) #Diagonal(vols)

    A = sparse(F♯) * G

    # easy because const on boxes, otherwise we need some quadrature rule
    L = Diagonal([ 1/Df(c) for (c,r) in S ]) * G

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

leg(x::AbstractFloat, R=256) = collectPl(x; norm=Val(:normalized), lmax = R-1)
leg(::Val{R}) where R = x -> leg(x, R)

nodes, weights = gausslegendre(1024)

function res_dmd(
        f = linear_gauss,
        nodes = nodes,
        weights = weights,
        basis = leg(Val(64)),#x -> [1; cospi.(2x .* (1:10)); sinpi.(2x .* (1:10))],
        xs = xs, 
        ys = ys
    )
    
    data = 2 .* f.( (nodes .- 1) ./ 2 ) .+ 1

    ΨX = reduce(hcat, basis.(nodes))'
    ΨY = reduce(hcat, basis.(data))'

    W = Diagonal(weights)

    G = ΨX' * W * ΨX
    A = ΨX' * W * ΨY
    L = ΨY' * W * ΨY

    G ≈ I && (G = I)
    L = (L + L') / 2

    function residual(z, G=G, A=A, L=L)
        λ = eigvals( inv(G) * (L - z * A' - z' * A + abs(z)^2 * G) )
        λ0 = λ[ argmin(abs.(λ)) ]
        real(λ0) < 0 && @warn "erroneous negative residual calculated" z calculated_residual=λ0
        sqrt(max(real(λ0), 0))
    end

    res = @showprogress broadcast(xs', ys) do x, y
        z = x + y * im
        if abs(z) > 1.2 && x < 1
            NaN
        else
            @exitsafe residual(z)
        end
    end

    λ = eigvals(A)

    return res, λ
end

r_ulam, λ_ulam = res_ulam()
r_dmd, λ_dmd = res_dmd()

levels = [0.001; 0.005; 0.01; 0.02; 0.05; 0:0.1:1;]

p1 = contour(
    xs, ys, r_ulam, 
    levels=levels, aspectratio=1., colormap=:rainbow, clabels=true
)
scatter!(λ_ulam, label="Ulam eigs", marker=:xcross)
scatter!(λ_dmd, label="EDMD eigs", marker=:cross)
plot!(cospi.(0:0.01:2), sinpi.(0:0.01:2), style=:dash, label="|z| = 1")

p2 = contour(
    xs, ys, r_dmd, 
    levels=levels, aspectratio=1., colormap=:rainbow, clabels=true
)
scatter!(λ_ulam, label="Ulam eigs", marker=:xcross)
scatter!(λ_dmd, label="EDMD eigs", marker=:cross)
plot!(cospi.(0:0.01:2), sinpi.(0:0.01:2), style=:dash, label="|z| = 1")
#scatter!(ComplexF64[((-0.8*exp(2π*im/9)) .^ (1:10)); ((-0.8*exp(-2π*im/9)) .^ (1:10))], marker=:xcross)

plot(p1, p2, size=(1200,600))

savefig("../talk_GAIO.jl/pseudospectrum_comparison_gauss.png")


anim = @animate for n_basis in [32, 64, 128, 256]
    n_points = 4*n_basis
    nodes, weights = gausslegendre(n_points)
    r_dmd, λ_dmd = res_dmd(mob, nodes, weights, leg(Val(n_basis)))
    p3 = contour(
        xs, ys, r_dmd, 
        levels=levels, aspectratio=1., colormap=:rainbow, clabels=true,
        title="$n_basis legendre polynomial basis"
    )
    scatter!(p3, λ_dmd, label="EDMD eigs", marker=:cross)
    plot!(p3, cospi.(0:0.01:2), sinpi.(0:0.01:2), style=:dash, label="|z| = 1")
end
gif(anim, fps=0.5)

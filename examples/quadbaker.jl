using GAIO
using StaticArrays, SparseArrays, LinearAlgebra, Arpack, ProgressMeter
using Plots

macro exitsafe(expr)
    return quote
        try
            $(esc(expr))
        catch err
            err isa InterruptException && rethrow()
            showerror(stdout, err)
            NaN
        end
    end
end

function quadbaker( (x,y) )
    if x < 1/2
        if y < 1/4
            ( x/4, 4y )
        elseif y < 1/2
            ( x/4 + 1/8, 4y - 1 )
        elseif y < 3/4
            ( x/4 + 1/4, 4y - 2 )
        else
            ( x/4 + 1/2, 4y - 3 )
        end
    else
        if y < 1/4
            ( (x - 1/2)/4 + 5/8, 4y )
        elseif y < 1/2
            ( (x - 1/2)/4 + 3/4, 4y - 1 )
        elseif y < 3/4
            ( (x - 1/2)/4 + 7/8, 4y - 2 )
        else
            ( (x - 1/2)/4 + 3/8, 4y - 3 )
        end
    end
end

jacobian_quadbaker(z) = @SMatrix [1/4 0; 0 4]

cen = rad = (0.5, 0.5)
domain = Box(cen, rad)
P = BoxPartition(domain, (128,128))
S = cover(P, :)

F = BoxMap(:grid, quadbaker, domain)
F♯ = TransferOperator(F, S, S)
S = F♯.domain # ordered

λ, ev, nconv = eigs(F♯, nev=32)
plot(ev[1])
plot(ev[2])

# easy because all same, otherwise use vector
vols = volume(first(S)) #[volume(Ai) for Ai in S]
G = I#UniformScaling(vols) #Diagonal(vols)

A = sparse(F♯)# * G

# easy because const==1, otherwise we need some kind of quadrature rule
L = I#G #./ det(jacobian_quadbaker((0,0))) 

function residual(z, G=G, A=A, L=L)
    ξ, _ = eigs( 
        #=inv(G) * =#(L - z * A' - z' * A + abs(z)^2 * G),
        which=:SM, nev=3, ritzvec=false,  tol=0#10*eps()
    )
    ξ0 = ξ[ argmin(abs.(ξ)) ]
    real(ξ0) < 0 && @warn "erroneous negative residual calculated" z calculated_residual=ξ0
    sqrt( max( real(ξ0), 0 ) )
end

ys = -1.05:0.05:1.05
xs = -1.05:0.05:1.2

res = @showprogress broadcast(xs', ys) do x, y
    z = x + y * im
    if (abs(z) > 1.1 && x < 1) || abs(z) > 1.25
        NaN
    else
        @exitsafe residual(z)
    end
end

xs2 = 0.49:0.005:0.51
ys2 = -0.01:0.005:0.01
res2 = @showprogress broadcast(xs2', ys2) do x, y
    @exitsafe residual(x + y * im)
end


levels = [0:0.05:1;]#[0.001, 0.05, 0.13, 0.16, 0.2, 0.3]

contour(
    xs, ys, #=log10.(=#res#=)=#, 
    levels=levels,#10 .^ (-2:0.01:0),#[0.001,0.01,0.1,0.3], 
    clims=(0,0.6),
    size=(900,700), color=:rainbow, aspectratio=1., clabels=true, 
    #title=""
)

contour!(
    xs2, ys2, res2, 
    levels=levels, 
    color=:rainbow, clabels=true
)

scatter!(λ, marker=:xcross, label="Ulam eigs")
plot!(
    cospi.(0:0.001:2)/3, sinpi.(0:0.001:2)/3, 
    style=:dash, label="|z| = 1/3"
)

plot!(
    cospi.(0:0.001:2), sinpi.(0:0.001:2), 
    style=:dash, label="|z| = 1"
)

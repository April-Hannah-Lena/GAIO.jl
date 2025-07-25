using GAIO
using StaticArrays
using Test

@testset "exported functionality" begin
    function f(u)   # the Baker transformation
        x, y = u
        if x < 0.5
            (2x, y/2)
        else
            (2x - 1, y/2 + 1/2)
        end
    end

    c = r = SA_F32[0.5, 0.5]
    domain = Box(c, r)

    F = BoxMap(:grid, f, domain)
    P = BoxGrid(domain, (16,16))

    S = cover(P, :)
    F♯ = TransferOperator(F, S, S)

    λ, μs, nconv = eigs(F♯, tol=100*eps(), v0=ones(size(F♯,2)))
    @test λ[1] ≈ 1

    μ = μs[1]
    u = values(μ)
    @test all( u .≈ sum(u) / length(u) )
end

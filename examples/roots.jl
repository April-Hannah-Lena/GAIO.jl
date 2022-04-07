using GAIO
using ForwardDiff
using StaticArrays

# domain (-40,40)^n, 3^n roots in domain
dim = 3

g(x) = 100 .* x .+ x .^ 2 .- x .^ 3 .- sum(x)
Dg = x -> ForwardDiff.jacobian(g, SVector{dim,Float64}(x))

center, radius = zeros(dim), 40*ones(dim)
P = BoxPartition(Box(center, radius))

R = cover_roots(g, Dg, P[:], steps=dim*8)

plot(R)
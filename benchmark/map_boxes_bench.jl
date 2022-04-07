ENV["JULIA_DEBUG"] = "all"
using GAIO
using StaticArrays, Base.Threads, MuladdMacro, SIMD, HostCPUFeatures
import GAIO.map_boxes, GAIO.point_to_key, GAIO.unsafe_point_to_ints, GAIO.key_to_box

function tuple_vgather(v::V, simd) where V<:AbstractVector{SV} where SV<:Union{NTuple{N,T}, StaticVector{N,T}} where {N,T}
    n = length(v)
    vr = reinterpret(T, v)
    idx = SIMD.Vec(ntuple(x -> N*(x-1), simd))
    vo = ntuple(i -> vr[idx + i], N)
    return vo
end

function tuple_vscatter(vo::SV) where SV<:Union{NTuple{N,SIMD.Vec{simd,T}}, <:StaticVector{N,SIMD.Vec{simd,T}}} where {N,T,simd}
    idx = SIMD.Vec(ntuple(x -> N*(x-1), Val(simd)))
    vr = Vector{T}(undef, N*simd)
    for i in 1:N
        vr[idx + i] = vo[i]
    end
    v = reinterpret(NTuple{N,T}, vr)
    return v
end

function tuple_vscatter(vo::SV) where SV<:Union{NTuple{N,SIMD.Vec{simd,Bool}}, <:StaticVector{N,SIMD.Vec{simd,Bool}}} where {N,simd} 
    vor = reinterpret.(Vec{simd,UInt8}, vo)
    idx = SIMD.Vec(ntuple(x -> N*(x-1), Val(simd)))
    vr = Vector{UInt8}(undef, N*simd)
    for i in 1:N
        vr[idx + i] = vor[i]
    end
    v = reinterpret(NTuple{N,Bool}, vr)
    return v
end

function GAIO.unsafe_point_to_ints(partition::BoxPartition, point::SV) where SV<:Union{NTuple{N,SIMD.Vec{simd,T}}, <:StaticVector{N,SIMD.Vec{simd,T}}} where {N,T,simd}
    x = (point .- partition.left) .* partition.scale
    x_ints = map(x) do xi
        convert(Vec{simd, Int}, trunc(xi))
    end
    return x_ints
end

function ints_to_key(partition::BoxPartition, x_ints)
    if any(x_ints .< zero(eltype(x_ints))) || any(x_ints .>= partition.dims)
        @debug "point does not lie in the domain" point partition.domain
        return nothing
    end
    key = sum(x_ints .* partition.dimsprod) + 1
    return key
end

function ints_to_key(partition::BoxPartition, x_ints::SV) where SV<:Union{NTuple{N,SIMD.Vec{simd,T}}, <:StaticVector{N,SIMD.Vec{simd,T}}} where {N,T,simd}
    in_bounds = all.(
        tuple_vscatter(
            ( x_ints .>= zero(T) ) .& ( x_ints .< partition.dims )
        )
    )
    key = NTuple{simd,T}(sum(x_ints .* partition.dimsprod) + 1)
    return key[in_bounds]
end

function GAIO.point_to_key(partition::BoxPartition, point)
    x_ints = unsafe_point_to_ints(partition, point)
    key = ints_to_key(partition, x_ints)
    bound = partition.dimsprod[end] * partition.dims[end]
    if key > bound
        @debug "key out of bounds"
        key = bound
    end
    return key
end
function GAIO.point_to_key(partition::BoxPartition, point::SV) where SV<:Union{NTuple{N,SIMD.Vec{simd,T}}, <:StaticVector{N,SIMD.Vec{simd,T}}} where {N,T,simd}
    x_ints = unsafe_point_to_ints(partition, point)
    key = ints_to_key(partition, x_ints)
    bound = partition.dimsprod[end] * partition.dims[end]
    for i in eachindex(key)
        if key[i] .> bound
            @debug "key out of bounds"
            key[i] = bound
        end
    end
    return key 
end

function GAIO.key_to_box(partition::BoxPartition{N,T}, key::M) where M <: Union{Int, NTuple{N, Int}} where {N,T}
    dims = size(partition)
    radius = partition.domain.radius ./ dims
    left = partition.domain.center .- partition.domain.radius
    #@show out_key=key
    center = left .+ radius .+ (2 .* radius) .* (CartesianIndices(dims)[key].I .- 1)
    return Box(center, radius)
end 


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
println("Standard Version")

function GAIO.map_boxes(g::SampledBoxMap{N,T,Val{:cpu}}, source::BoxSet) where {N,T}
    P, keys = source.partition, collect(source.set)
    image = fill( Set{eltype(keys)}(), nthreads() )
    @threads for key in keys
        box = key_to_box(P, key)
        c, r = box.center, box.radius
        points = g.domain_points(c, r)
        for p in points
            fp = @muladd p .* r .+ c
            fp = g.map(fp)
            hit = point_to_key(P, fp)
            if !isnothing(hit)
                push!(image[threadid()], hit)
            end
        end
    end
    b = BoxSet(P, union(image...))
    #@show b.set
    return b
end

G = BoxMap(F, P, :cpu)
Y = G(P[x])
W = unstable_set!(G, P[x])
for _ in 1:10
    @time unstable_set!(G, P[x])
end

# -----------------------------------------
println("Partial cpu version")

function GAIO.map_boxes(g::SampledBoxMap{N,T,Val{:cpu}}, source::BoxSet) where {N,T}
    P, keys = source.partition, collect(source.set)
    image = fill( Set{eltype(keys)}(), nthreads() )
    test_points = g.domain_points(P.domain.center, P.domain.radius)
    n, simd = length(test_points), pick_vector_width(T)
    @assert n%simd==0 "Number of test points $n is not divisible by SIMD capability $(simd)"
    @threads for key in keys
        box = key_to_box(P, key)
        c, r = box.center, box.radius
        for i in 0:simd:n-simd
            points = tuple_vgather(view(test_points, i+1:i+simd), simd)
            points = @muladd points .* r .+ c
            points = g.map(points)
            for p in tuple_vscatter(points)
                hit = point_to_key(P, p)
                if !isnothing(hit)
                    push!(image[threadid()], hit)
                end
            end
        end
    end
    return BoxSet(P, union(image...))
end 

G = BoxMap(F, P, :cpu)
Y = G(P[x])
W = unstable_set!(G, P[x])
for _ in 1:10
    @time unstable_set!(G, P[x])
end

# -----------------------------------------
println("Full cpu version")

function GAIO.map_boxes(g::SampledBoxMap{N,T,Val{:cpu}}, source::BoxSet) where {N,T}
    P, keys = source.partition, collect(source.set)
    image = fill( Set{eltype(keys)}(), nthreads() )
    test_points = g.domain_points(P.domain.center, P.domain.radius)
    n, simd = length(test_points), pick_vector_width(T)
    @assert n%simd==0 "Number of test points $n is not divisible by SIMD capability $(simd)"
    @threads for key in keys
        box = key_to_box(P, key)
        c, r = box.center, box.radius
        for i in 0:simd:n-simd
            points = tuple_vgather(view(test_points, i+1:i+simd), simd)
            points = @muladd points .* r .+ c
            points = g.map(points)
            keys = point_to_key(P, points)
            push!(image[threadid()], keys...)
        end
    end
    return BoxSet(P, union(image...))
end 

G = BoxMap(F, P, :cpu)
Y = G(P[x])
W = unstable_set!(G, P[x])
for _ in 1:10
    @time unstable_set!(G, P[x])
end

#plot(W)
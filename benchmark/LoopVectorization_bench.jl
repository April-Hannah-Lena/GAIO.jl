ENV["JULIA_DEBUG"] = "all"

using GAIO
using Test
using LoopVectorization, MuladdMacro, Base.Threads, StaticArrays

a, b = 1.4, 0.3
function f(x)
    return (1 - a*(x[1])^2 + x[2], b*(x[1]))
end
function f2(x)
    dx = similar(x)
    @turbo for i in 0 : 2 : length(x) - 2
        dx[i+1] = 1 - a*(x[i+1])^2 + x[i+2]
        dx[i+2] = b*(x[i+1])
    end
    return dx
end


center, radius, no_of_boxes = (0, 0), (3, 3), (256, 256)
P = BoxPartition(Box(center, radius), no_of_boxes)

N, T = 2, Float64
no_of_points = 64
points = [ SVector{N,T}(2.0*rand(T,N).-1.0) for _ = 1:no_of_points ];
points_cache = deepcopy(points);

# simple boxmap
F = PointDiscretizedMap(f, P.domain, points)
#F = BoxMap(f, P, no_of_points=60)

# boxmap with the same test points
F2 = PointDiscretizedMap(f2, P.domain, points, Val(:cpu))
#F2 = SampledBoxMap(f2, F.domain, F.domain_points, F.image_points, Val(:cpu))

# random point in the domain
x = SVector{N,T}((rand(T,N) .- 0.5) .* radius .+ center)
@test all( f(x) .≈ f2(x) )
@test all( f(x) .== f2(x) )
box = key_to_box(P, point_to_key(P, x))
c, r = box.center, box.radius
@test all( F.domain_points(c, r) .== F2.domain_points(c, r) )

import GAIO.map_boxes
function GAIO.map_boxes(g::SampledBoxMap{N,T,Nothing}, source::BoxSet) where {N,T}
    @debug "using no accel"
    P, keys = source.partition, collect(source.set)
    image = fill( Set{eltype(keys)}(), nthreads() )
    in_points = fill( SVector{N,T}[], nthreads() )
    out_points = fill( SVector{N,T}[], nthreads() )
    @threads for key in keys
        box = key_to_box(P, key)
        c, r = box.center, box.radius
        points = g.domain_points(c, r)
        for p in points
            fp = @muladd p .* r .+ c
            push!(in_points[threadid()], fp)
            fp = g.map(fp)
            push!(out_points[threadid()], fp)
            hit = point_to_key(P, fp)
            if hit !== nothing
                push!(image[threadid()], hit)
            end
        end
    end
    return BoxSet(P, union(image...)), vcat(in_points...), vcat(out_points...)
end 

function GAIO.map_boxes(g::SampledBoxMap{N,T,Val{:cpu}}, source::BoxSet) where {N,T}
    @debug "using :cpu accel"
    P, keys = source.partition, collect(source.set)
    image = fill( Set{eltype(keys)}(), nthreads() )
    in_points = fill( SVector{N,T}[], nthreads() )
    out_points = fill( SVector{N,T}[], nthreads() )
    @threads for key in keys
        box = key_to_box(P, key)
        c, r = box.center, box.radius
        points = deepcopy(g.domain_points(c, r))
        points_vec = reinterpret(T, points)
        @debug " " type=typeof(points_vec) vec_size=size(points_vec) rest_c=size(points_vec).%size(c) rest_r=size(points_vec).%size(r)
        @turbo for i in 0:N:length(points_vec)-N, j in 1:N
            points_vec[i+j] = @muladd points_vec[i+j] * r[j] + c[j]
        end
        push!(in_points[threadid()], reinterpret(SVector{N,T}, points_vec)...)
        points_vec .= g.map(points_vec)
        points = reinterpret(SVector{N,T}, points_vec)
        push!(out_points[threadid()], points...)
        for p in points
            hit = point_to_key(P, p)
            if hit !== nothing
                push!(image[threadid()], hit)
            end
        end
    end
    return BoxSet(P, union(image...)), vcat(in_points...), vcat(out_points...)
end 


set1, in_points1, out_points1 = map_boxes(F, P[x])
set2, in_points2, out_points2 = map_boxes(F2, P[x])

# test if all points are mapped the same
@test all( in_points1 .≈ in_points2 )
@test all( in_points1 .== in_points2 )
@test all( out_points1 .≈ out_points2 )
@test all( out_points1 .== out_points2 )
# test if the resulting boxsets are the same
@test issetequal(set1.set, set2.set)
# test if somewhere an object that shouldn't be changed got changed
@test all( points .== points_cache )
@test all( F.domain_points(c, r) .== F2.domain_points(c, r) )
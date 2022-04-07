using StaticArrays, Base.Threads, MuladdMacro, SIMD, HostCPUFeatures
import GAIO.map_boxes, GAIO.point_to_key, GAIO.unsafe_point_to_ints, GAIO.key_to_box, GAIO.ints_to_key

#function tuple_vgather(v::V, simd) where V<:AbstractVector{SV} where SV<:Union{NTuple{N,T}, <:StaticVector{N,T}} where {N,T}
function tuple_vgather(v::V, simd) where V<:AbstractVector{NTuple{N,T}} where {N,T}
    n = length(v)
    vr = reinterpret(T, v)
    idx = SIMD.Vec(ntuple(x -> N*(x-1), simd))
    vo = ntuple(i -> vr[idx + i], N)
    return vo
end

#function tuple_vscatter(vo::SV) where SV<:Union{NTuple{N,SIMD.Vec{simd,T}}, <:StaticVector{N,SIMD.Vec{simd,T}}} where {N,T,simd}
function tuple_vscatter(vo::NTuple{N,SIMD.Vec{simd,T}}) where {N,T,simd}
    idx = SIMD.Vec(ntuple(x -> N*(x-1), Val(simd)))
    vr = Vector{T}(undef, N*simd)
    for i in 1:N
        vr[idx + i] = vo[i]
    end
    v = reinterpret(NTuple{N,T}, vr)
    return v
end

#function tuple_vscatter(vo::SV) where SV<:Union{NTuple{N,SIMD.Vec{simd,Bool}}, <:StaticVector{N,SIMD.Vec{simd,Bool}}} where {N,simd} 
function tuple_vscatter(vo::NTuple{N,SIMD.Vec{simd,Bool}}) where {N,simd}
    vor = reinterpret.(Vec{simd,UInt8}, vo)
    idx = SIMD.Vec(ntuple(x -> N*(x-1), Val(simd)))
    vr = Vector{UInt8}(undef, N*simd)
    for i in 1:N
        vr[idx + i] = vor[i]
    end
    v = reinterpret(NTuple{N,Bool}, vr)
    return v
end

#function GAIO.unsafe_point_to_ints(partition::BoxPartition, point::SV) where SV<:Union{NTuple{N,SIMD.Vec{simd,T}}, <:StaticVector{N,SIMD.Vec{simd,T}}} where {N,T,simd}
function GAIO.unsafe_point_to_ints(partition::BoxPartition, point::NTuple{N,SIMD.Vec{simd,T}}) where {N,T,simd}
    x = (point .- partition.left) .* partition.scale
    x_ints = map(x) do xi
        convert(Vec{simd, Int}, trunc(xi))
    end
    return x_ints
end

#function GAIO.ints_to_key(partition::BoxPartition, x_ints::SV) where SV<:Union{NTuple{N,SIMD.Vec{simd,T}}, <:StaticVector{N,SIMD.Vec{simd,T}}} where {N,T,simd}
function GAIO.ints_to_key(partition::BoxPartition, x_ints::NTuple{N,SIMD.Vec{simd,T}}) where {N,T,simd}
    in_bounds = all.(
        tuple_vscatter(
            ( x_ints .>= zero(T) ) .& ( x_ints .< partition.dims )
        )
    )
    key = NTuple{simd,T}(sum(x_ints .* partition.dimsprod) + 1)
    return key[in_bounds]
end

#function GAIO.point_to_key(partition::BoxPartition, point::SV) where SV<:Union{NTuple{N,SIMD.Vec{simd,T}}, <:StaticVector{N,SIMD.Vec{simd,T}}} where {N,T,simd}
function GAIO.point_to_key(partition::BoxPartition, point::NTuple{N,SIMD.Vec{simd,T}}) where {N,T,simd}
    x_ints = unsafe_point_to_ints(partition, point)
    key = ints_to_key(partition, x_ints)
    bound = partition.dimsprod[end] * partition.dims[end]
    for i in eachindex(key)
        if key[i] .> bound
            @debug "key out of bounds" key bound
            key[i] = bound
        end
    end
    return key 
end

function GAIO.map_boxes(g::SampledBoxMap{N,T,Val{:cpu}}, source::BoxSet) where {N,T}
    P, keys = source.partition, collect(source.set)
    image = fill( Set{eltype(keys)}(), nthreads() )
    test_points = g.domain_points(P.domain.center, P.domain.radius)
    n, simd = length(test_points), pick_vector_width(T)
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

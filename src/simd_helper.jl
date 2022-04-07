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
    vor = reinterpret.(SIMD.Vec{simd,UInt8}, vo)
    idx = SIMD.Vec(ntuple(x -> N*(x-1), Val(simd)))
    vr = Vector{UInt8}(undef, N*simd)
    for i in 1:N
        vr[idx + i] = vor[i]
    end
    v = reinterpret(NTuple{N,Bool}, vr)
    return v
end

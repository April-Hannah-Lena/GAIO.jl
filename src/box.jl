# abstract type AbstractBoxPartition{B <: Box} end

"""
A generalized box with
`center`:   Tuple where the box's center is located
`radius`:   Tuple of radii, length of the box in each dimension

"""
struct Box{N,T <: AbstractFloat}
    center::NTuple{N,T}
    radius::NTuple{N,T}

    function Box(center, radius)
        N = length(center)

        if length(radius) != N
            throw(DimensionMismatch("Center Tuple and radius Tuple must have same length ($N)"))
        end

        if any(x -> x <= 0, radius)
            throw(DomainError(radius, "radius must be positive in every component"))
        end
        
        T = promote_type(eltype(center), eltype(radius))
        if !(T <: AbstractFloat)
            T = Float64
        end

        return new{N,T}(NTuple{N,T}(center), NTuple{N,T}(radius))
    end
end

Base.in(point, box::Box) = all(box.center .- box.radius  .<=  point  .<  box.center .+ box.radius)

volume(box::Box) = prod(2 .* box.radius)

function Base.show(io::IO, box::Box) 
    print(io, "Box: center = $(box.center), radii = $(box.radius)")
end

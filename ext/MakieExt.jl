module MakieExt

using GAIO, Makie, GeometryBasics, StaticArrays
    
"""
    plot(boxset::BoxSet)
    plot(boxmeas::BoxMeasure)
    plot!(boxset::BoxSet)
    plot!(boxmeas::BoxMeasure)

Plot a `BoxSet` or `BoxMeasure`. 

## Special Attributes:

`projection = x -> x[1:3]`
If the dimension of the system is larger than 3, use this function to project to 3-d space.

`colormap = :default`
Colormap used for plotting `BoxMeasure`s values.

`marker = HyperRectangle(GeometryBasics.Vec3f(0), GeometryBasics.Vec3f(1))`
The marker for an individual box. Only works if using Makie for plotting. 

All other attributes are taken from MeshScatter.

"""
@recipe PlotBoxes begin
    Makie.documented_attributes(Makie.MeshScatter)...

    marker = HyperRectangle(
        GeometryBasics.Vec3f(0),
        GeometryBasics.Vec3f(1),
    )
    projection = nothing 
end

Makie.preferred_axis_type(::PlotBoxes) = Axis3

function Makie.plot!(boxes::PlotBoxes{<:Tuple{<:BoxSet{GAIO.Box{N,T}}}}) where {N,T}

    boxset = boxes[1][]
    d = min(N, 3)
    if isnothing(boxes.projection[])
        boxes.projection[] = x -> x[1:d]
    end
    q = boxes.projection[]

    center = Vector{GeometryBasics.Vec{d, Float32}}(undef, length(boxset))
    radius = Vector{GeometryBasics.Vec{d, Float32}}(undef, length(boxset))

    for (i, box) in enumerate(boxset)
        center[i] = q(box.center)
        radius[i] = q(box.radius) .* 1.9
    end

    Makie.meshscatter!(
        boxes, 
        boxes.attributes, 
        center, 
        markersize = radius
    )
end

function Makie.plot!(boxes::PlotBoxes{<:Tuple{<:BoxMeasure{GAIO.Box{N,T}}}}) where {N,T}

    boxmeas = boxes[1][]
    d = min(N, 3)
    if isnothing(boxes.projection[])
        boxes.projection[] = x -> x[1:d]
    end
    q = boxes.projection[]

    center = Vector{GeometryBasics.Vec{d, Float32}}(undef, length(boxmeas))
    radius = Vector{GeometryBasics.Vec{d, Float32}}(undef, length(boxmeas))
    colors = Vector{Float32}(undef, length(boxmeas))

    for (i, (box, value)) in enumerate(boxmeas)
        center[i] = q(box.center)
        radius[i] = q(box.radius) .* 1.9
        colors[i] = value
    end

    haskey(boxes.kw, :color)  ||  Makie.update!(boxes, color=colors)
    #boxes.colorrange[] = extrema(colors)

    Makie.meshscatter!(
        boxes, 
        boxes.attributes,
        center, 
        markersize = radius, 
    )
end

function Makie.plot!(boxes::PlotBoxes{<:Tuple{<:BoxMeasure{GAIO.Box{2,T}}}}) where {T}

    boxmeas = boxes[1][]

    center = Vector{GeometryBasics.Vec{3, Float32}}(undef, 2*length(boxmeas))
    radius = Vector{GeometryBasics.Vec{3, Float32}}(undef, 2*length(boxmeas))

    for (i, (box, value)) in enumerate(boxmeas)
        center[2*i-1] = SVector{3,Float32}(box.center..., 0.)
        center[2*i]   = SVector{3,Float32}(box.center..., value)
        radius[2*i-1] = SVector{3,Float32}(box.radius..., minimum(box.radius))
        radius[2*i]   = radius[2*i-1]
    end

    boxes.colorrange[] = extrema(x -> x[3], center)

    Makie.meshscatter!(
        boxes, 
        boxes.attributes, 
        center, 
        markersize  = radius
    )
end

function Makie.plot!(boxes::PlotBoxes{<:Tuple{<:BoxMeasure{GAIO.Box{1,T}}}}) where {T}

    boxmeas = boxes[1][]

    height = Vector{Float32}(undef, 2*length(boxmeas))
    center = Vector{Float32}(undef, 2*length(boxmeas))
    radius = Vector{Float32}(undef, 2*length(boxmeas))

    for (i, (box, value)) in enumerate(boxmeas)
        height[2*i-1] = 0.
        height[2*i]   = value
        center[2*i-1] = box.center[1]
        center[2*i]   = center[2*i-1]
        radius[2*i-1] = box.radius[1] * 1.9
        radius[2*i]   = radius[2*i-1]
    end

    boxes.colorrange[] = extrema(height)

    Makie.linesegments!(
        boxes, 
        boxes.attributes,
        center,
        height,
        linewidth  = radius .* 1f3
    )
end

Makie.plottype(::Union{BoxSet,BoxMeasure}) = PlotBoxes

function Makie.convert_arguments(::Makie.PointBased, coords::AbstractVector{<:Complex})
    #Float32.(real.(coords)), Float32.(imag.(coords))
    (map(x -> Point2f0(real(x), imag(x)), coords),)
end

function Makie.convert_arguments(::Makie.PointBased, coords::AbstractVector{<:Complex}, heights::AbstractVector{<:Real})
    #Float32.(real.(coords)), Float32.(imag.(coords)), Float32.(heights)
    (map((x,y) -> Point3f0(real(x), imag(x), y)),)
end

Makie.plottype(::AbstractVector{<:Complex}) = Scatter

end # module

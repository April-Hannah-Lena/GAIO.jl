function MatrixNetworks.MatrixNetwork(F♯::TransferOperator)
    @assert F♯.domain == F♯.codomain
    adj = copy(F♯.mat')
    MatrixNetwork(size(adj, 1), getcolptr(adj), rowvals(adj), nonzeros(adj))
end

function MatrixNetworks.scomponents(F♯::TransferOperator)
    scomponents(MatrixNetwork(F♯))
end

function istrivial(condensation_graph, sizes, vertex)
    sizes[vertex] == 1 && iszero(condensation_graph[vertex, vertex])
end

"""
Given a `Strong_components_output` from `MatrixNetworks` (in particular 
the component map), compute a second map on the vertices of the 
condensation graph to the vertices of the morse graph. Vertices of
the condensation graph which do not correspond to morse sets, get
sent to the (arbitrary) vertex index 0.

     origninal         condensation        morse
     graph             graph               graph

        * ──────┐
                │
        * ──────┴───────→ * ───────────────→ *

        * ──────────────→ * ───┐     ┌─────→ *
                               │     │
        * ──────────────→ * ───┴─────┼────→  0
                                     │
        * ─────┬────────→ * ─────────┘
               │
        * ─────┤
               │
        * ─────┘

        ⋮ ==============⟹ ⋮ ==============⟹ ⋮
            condensation        morse
            map                 map

.
"""
function morse_map(strong_components::Strong_components_output)
    strong_components_enrich = enrich(strong_components)
    
    sizes = strong_components.sizes
    condensation_graph = strong_components_enrich.transitive_map
    vertices = 1:size(condensation_graph)[1]
    
    morse_map = spzeros(Int, length(vertices))
    nontrivial_count = 0
    
    for vertex in vertices
        if !istrivial(condensation_graph, sizes, vertex)
            morse_map[vertex] = (nontrivial_count += 1)
        end
    end
    
    return morse_map
end

"""
Concatenation of the condensation map and morse map. 
See `morse_map`
"""
function morse_component_map(strong_components::Strong_components_output, morse_map)
    component_map = strong_components.map
    vertices = 1:size(strong_components.A)[1]
    morse_comp_map = spzeros(Int, length(vertices))
    for vertex in vertices
        ind = morse_map[component_map[vertex]]
        if !iszero(ind)
            morse_comp_map[vertex] = ind
        end
    end

    return morse_comp_map
end

function morse_component_map(strong_components::Strong_components_output)
    morse_map_ = morse_map(strong_components)
    morse_component_map(strong_components, morse_map_)
end

#=
function morse_component_map(F♯::TransferOperator)
    morse_component_map(scomponents(F♯))
end

function morse_component_map(F::BoxMap, B::BoxSet)
    morse_component_map(TransferOperator(F, B, B))
end
=#

"""
Given a `strong_components_output` from `MatrixNetworks` (in particular 
the component map) as well as the morse map (see `morse_map`), compute 
the adjacency matrix for the morse graph. 
"""
function morse_adjacencies(strong_components::Strong_components_output, morse_map)
    strong_components_enrich = enrich(strong_components)
    sizes = strong_components.sizes
    condensation_graph = strong_components_enrich.transitive_map

    n = maximum(morse_map)  # size of morse graph
    morse_adjacencies = spzeros(Int, n, n)
    nontrivials = ( morse_map .> 0 )
    
    vertices = 1:size(condensation_graph)[1]
    for vertex in vertices
        if !istrivial(condensation_graph, sizes, vertex)
            dists, _ = bfs(condensation_graph, vertex)
            morse_vertex = morse_map[vertex]
            morse_adjacencies[morse_vertex, :] .+= ( dists[nontrivials] .> -1 )
        end
    end

    return morse_adjacencies
end

#=
function morse_adjacencies(strong_components::Strong_components_output)
    morse_map_ = morse_map(strong_components)
    morse_adjacencies(strong_components, morse_map_)
end

function morse_adjacencies(F♯::TransferOperator)
    morse_adjacencies(scomponents(F♯))
end

function morse_adjacencies(F::BoxMap, B::BoxSet)
    morse_adjacencies(TransferOperator(F, B, B))
end
=#

"""
Given a `BoxMap`, a domain `BoxSet` (together interpreted 
as a transfer graph `G`), 
compute the boxes corresponding to the vertices 
of the morse graph. These vertices are precisely the 
chain recurrent components of `G`
"""
function morse_tiles(
        domain::BoxSet{R,Q,S}, morse_component_map::AbstractVector;
        dicttype=OrderedDict{keytype(Q),Int}
    ) where {R,Q,S<:OrderedSet}

    tiles = BoxMeasure(domain.partition, dicttype())

    for (key, comp) in zip(domain.set, morse_component_map)
        if !iszero(comp)
            tiles[key] = comp
        end
    end

    return tiles
end

#=
function morse_tiles(
        domain::BoxSet, strong_components::Strong_components_output
    )
    
    morse_component_map_ = morse_component_map(strong_components)
    morse_tiles(domain, morse_component_map_)
end

function morse_tiles(
        F♯::TransferOperator, strong_components::Strong_components_output
    )

    morse_tiles(F♯.domain, strong_components)
end
=#

function morse_tiles(F::BoxMap, B::BoxSet)
    F♯ = TransferOperator(F, B, B)
    strong_components = scomponents(F♯)
    morse_component_map_F♯ = morse_component_map(strong_components)
    morse_tiles(F♯.domain, morse_component_map_F♯)
end

#=
function morse_tiles(F♯::TransferOperator)
    strong_components = scomponents(F♯)
    morse_tiles(F♯, strong_components)
end

function morse_tiles(F::BoxMap, B::BoxSet)
    morse_tiles(TransferOperator(F, B, B))
end
=#

"""
Given a `BoxMap` and a domain `BoxSet` (together interpreted 
as a transfer graph), 
compute the adjacency matrix for the mose graph as well as 
the boxes representing the vertices for the morse graph. 
"""
function morse_adjacencies_and_tiles(F::BoxMap, B::BoxSet)
    F♯ = TransferOperator(F, B, B)
    adj = MatrixNetwork(F♯)
    strong_components = scomponents(adj)

    morse_map_F♯ = morse_map(strong_components)
    morse_adjacencies_F♯ = morse_adjacencies(strong_components, morse_map_F♯)
    
    morse_component_map_F♯ = morse_component_map(strong_components, morse_map_F♯)
    morse_tiles_F♯ = morse_tiles(F♯.domain, morse_component_map_F♯)

    return morse_adjacencies_F♯, morse_tiles_F♯
end

#=
function morse_adjacencies_and_tiles(F♯::TransferOperator)
    adj = MatrixNetwork(F♯)
    strong_components = scomponents(adj)

    morse_map_F♯ = morse_map(strong_components)
    morse_adjacencies_F♯ = morse_adjacencies(strong_components, morse_map_F♯)
    
    morse_component_map_F♯ = morse_component_map(strong_components, morse_map_F♯)
    morse_tiles_F♯ = morse_tiles(F♯.domain, morse_component_map_F♯)

    return morse_adjacencies_F♯, morse_tiles_F♯
end

function morse_adjacencies_and_tiles(F::BoxMap, B::BoxSet)
    morse_adjacencies_and_tiles(TransferOperator(F, B, B))
end
=#

function morse_graph(args...)
    @warn """
    `morse_graph` requires the package `MetaGraphsNext`. 
    If you have not loaded this, please do so. If you have, 
    then the MethodError is caused by a more specific problem. 
    """
    throw(MethodError(morse_graph, tyeof.(args)))
end

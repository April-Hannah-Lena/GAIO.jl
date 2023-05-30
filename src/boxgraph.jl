"""
    Graph(gstar::TransferOperator) -> BoxGraph
    Graph(g::BoxMap, boxset::BoxSet) = Graph(TransferOperator(g, boxset, boxset))

Directed Graph representation of a `TransferOperator`. The 
boxes in a `BoxSet` are enumerated as in TransferOperator. 
This means if the `domain`, `codomain` are taken from a 
`TranferOperator`, then the graph vertices are numbered 
`1 .. length(domain ∪ codomain)`. 

A directed edge exists from the i'th box `b_i` to the j'th 
box `b_j` if the BoxMap `g` has `b_j ∩ g⁻¹(b_i) ≠ ∅`. 
Equivalently, 
```julia
has_edge(g::BoxGraph, i, j) = !iszero( Matrix(g.gstar)[j,i] )
```

`Graphs.jl` operations like 
```julia
vertices, edges, weights, inneighbors, outneighbors, # etc...
```
are supported. Algorithms in `Graphs.jl` 
should work "out of the box", but will return whatever `Graphs.jl` 
returns by default. To convert a (integer) vertex index from the 
graph into a box index from the partition, one can call 
```julia
BoxSet(boxgraph, graph_index_or_indices)
```
If you would like to see specific behavior 
implemented, please open an issue! 

## Implementation details, not important for use:

We want to turn a matrix representation 
```julia
        domain -->
codomain  .   .   .   .   .
    |     .   .   .   .   .
    |     .   .   .   .   .
    v     .   .  mat  .   .
          .   .   .   .   .
          .   .   .   .   .
```
into a graph representation 
```julia
  domain ∪ codomain
  .---------.   .
 / \\       /   /
.   .-----.---.
```
!! efficiently !!

Julia's Graphs package only allows integer-indexed
vertices so we need to enumerate domain ∪ codomain. 
To do this, we enumerate the domain, then skip 
the boxes in the codomain which are already in the 
domain, then continue enumerating the rest of the 
codomain. 

We therefore permute the row indices of the weight 
matrix so that the skipped elements of the codomain
come first. 
"""
struct BoxGraph{B,T,P<:TransferOperator{B,T}} <: Graphs.AbstractSimpleGraph{Int}
    gstar::P
    n_intersections::Int
end

function BoxGraph(gstar::TransferOperator)
    gstar.codomain === gstar.domain && return BoxGraph(gstar, length(gstar.domain))
    # permute the row indices so that we can skip already identified boxes
    cut = intersect!(copy(gstar.domain), gstar.codomain)
    n = length(cut)
    inds = [key_to_index(gstar.codomain, key) for key in cut.set]
    gstar.codomain = union!(cut, gstar.codomain)
    gstar.mat[[1:n; inds], :] .= gstar.mat[[inds; 1:n], :]
    return BoxGraph(gstar, n)
end

Graphs.Graph(gstar::TransferOperator) = BoxGraph(gstar)
Graphs.Graph(g::BoxMap, boxset::BoxSet) = Graph(TransferOperator(g, boxset, boxset))

function Base.show(io::IO, g::BoxGraph{B,T,P}) where {B,T,P}
    print(io, "{$(nv(g)), $(ne(g))} directed simple $Int graph representation of TransferOperator")
end

function Base.show(io::IO, ::MIME"text/plain", g::BoxGraph{B,T,P}) where {B,T,P}
    print(io, "{$(nv(g)), $(ne(g))} directed simple $Int graph representation of TransferOperator")
end

function row_to_index(g::BoxGraph, row)
    g.gstar.domain === g.gstar.codomain && return row
    if row ≤ g.n_intersections
        u = index_to_key(g.gstar.codomain, row)
        j = key_to_index(g.gstar.domain, u)    
    else
        m, n = size(g.gstar)
        j = row + n - g.n_intersections
    end
    return j
end

function index_to_row(g::BoxGraph, j)
    g.gstar.domain === g.gstar.codomain && return j
    m, n = size(g.gstar)
    if j ≤ n
        u = index_to_key(g.gstar.domain, j)
        row = key_to_index(g.gstar.codomain, u)
    else
        row = j - n + g.n_intersections
    end
    return row
end

# partition-key to vertex-index
function key_to_index(g::BoxGraph, u)
    i = key_to_index(g.gstar.domain, u)
    !isnothing(i) && return i
    j = row_to_index(g, key_to_index(g.gstar.codomain, u))
    return j
end

# vertex-index to partition-key
function index_to_key(g::BoxGraph, j)
    m, n = size(g.gstar)
    j ≤ n && return index_to_key(g.gstar.domain, j)
    return index_to_key(g.gstar.codomain, index_to_row(g, j))
end

Base.eltype(::BoxGraph{B,T}) where {B,T} = B
Graphs.edgetype(::BoxGraph) = Graphs.SimpleEdge{Int}
Graphs.is_directed(::Type{<:BoxGraph}) = true
Graphs.is_directed(::BoxGraph) = true
Graphs.weights(g::BoxGraph) = g.gstar.mat'

Graphs.nv(g::BoxGraph) = sum(size(g.gstar.mat)) - g.n_intersections
Graphs.vertices(g::BoxGraph) = 1:nv(g)
Graphs.has_vertex(g::BoxGraph, i::Integer) = 1 ≤ i ≤ nv(g)
Graphs.has_vertex(g::BoxGraph, key) = has_vertex(g, key_to_index(g, key))

function Graphs.edges(g::BoxGraph)
    return (
        Graphs.SimpleEdge{Int}(i, row_to_index(g, j))
        for (j,i,w) in zip(findnz(g.gstar.mat)...)
    )
end
        
function Graphs.has_edge(g::BoxGraph, i::Integer, j::Integer)
    ĵ = index_to_row(g, j)
    v = g.gstar.mat[ĵ, i]
    return !iszero(v)
end

Graphs.ne(g::BoxGraph) = length(nonzeros(g.gstar.mat))
Graphs.has_edge(g::BoxGraph, u, v) = has_edge(g, key_to_index(g, u), key_to_index(g, v))

Graphs.SimpleGraphs.badj(g::BoxGraph, v) = collect(inneighbors(g, v))
Graphs.SimpleGraphs.badj(g::BoxGraph) = [Graphs.SimpleGraphs.badj(g, v) for v in Graphs.vertices(g)]
Graphs.SimpleGraphs.fadj(g::BoxGraph, v) = collect(outneighbors(g, v))
Graphs.SimpleGraphs.fadj(g::BoxGraph) = [Graphs.SimpleGraphs.fadj(g, v) for v in Graphs.vertices(g)]

function Graphs.LinAlg.adjacency_matrix(g::BoxGraph{B,T}) where {B,T} 
    g.gstar.domain === g.gstar.codomain && return copy(g.gstar.mat')
    
    m, n = size(g.gstar)
    set = g.gstar.domain ∪ g.gstar.codomain
    N = length(set)
    mat = spzeros(T, N, N)
    mat[n + g.n_intersections + 1 : end, 1:n] .= g.gstar.mat[g.n_intersections + 1 : end, :]

    cut = g.gstar.codomain ∩ g.gstar.domain
    dom_inds = [key_to_index(g.gstar.domain, key) for key in cut.set]
    codom_inds = [key_to_index(g.gstar.codomain, key) for key in cut.set]
    mat[dom_inds, 1:n] .= g.gstar.mat[codom_inds, :]

    return mat'
end

#Graphs.outneighbors(g::BoxGraph, v::Integer) = findall(!iszero, g.gstar.mat[:, v])
# efficiently find the nonzero rows corresponding to a column
function Graphs.outneighbors(g::BoxGraph, v::Integer)
    m, n = size(g.gstar)
    rows = rowvals(g.gstar.mat)
    # take nzrange for column or empty range if v > n, i.e. transfers out of v not calulated.
    # we do it this way to ensure that the result is type stable
    iterrange = 1 ≤ v ≤ n ? nzrange(g.gstar.mat, v) : (1:0)

    return Int[ row_to_index(g, row) for row in @view(rows[iterrange]) ]
end

#Graphs.inneighbors(g::BoxGraph,  u::Integer) = findall(!iszero, g.gstar.mat[u, :])
# efficiently find the nonzero columns related to a row 
function Graphs.inneighbors(g::BoxGraph, u::Integer)
    rows = rowvals(g.gstar.mat)
    colptr = SparseArrays.getcolptr(g.gstar.mat)
    j = index_to_row(g, u)
    return Int[ findfirst(>(i), colptr) - 1 for (i, row) in enumerate(rows) if row == j ]
end

function union_strongly_connected_components(g::BoxGraph)
    P = g.gstar.domain.partition

    sccs = Graphs.strongly_connected_components(Graphs.IsDirected{typeof(g)}, g)
    connected_vertices = OrderedSet{keytype(typeof(P))}()

    for scc in sccs
        if length(scc) > 1 || has_edge(g, scc[1], scc[1])
            union!(
                connected_vertices, 
                (index_to_key(g, i) for i in scc)
            )
        end
    end
    
    return BoxSet(P, connected_vertices)
end

"""
    BoxSet(boxgraph, graph_index_or_indices) -> BoxSet

Construct a BoxSet from some 
index / indices of vertices in a BoxGraph. 
"""
function BoxSet(g::BoxGraph{B,T,P}, inds) where {B,T,B1,P1,R,S<:BoxSet{B1,P1,R},P<:TransferOperator{B,T,S}} # where {B,T,Q,R,S<:BoxSet{B,Q,R},P<:TransferOperator{B,T,S}}
    keys = (index_to_key(g, i) for i in inds)
    return BoxSet(g.gstar.domain.partition, R(keys))
end

function BoxSet(g::BoxGraph{B,T,P}, ind::Integer) where {B,T,B1,P1,R,S<:BoxSet{B1,P1,R},P<:TransferOperator{B,T,S}} # where {B,T,Q,R,S<:BoxSet{B,Q,R},P<:TransferOperator{B,T,S}}
    BoxSet(g, (ind,))
end

function Graphs.SimpleDiGraph(gstar::TransferOperator)
    G = Graph(gstar)
    adj = adjacency_matrix(G)
    SimpleDiGraph(adj)
end

Graphs.SimpleDiGraph(g::BoxGraph) = SimpleDiGraph(g.gstar)

function right_resolving_representation(P::PT1, G::BoxGraph) where {PT1<:AbstractBoxPartition}
    R = MetaGraph(
        DiGraph(); 
        label_type = Set{Int}, # hypernodes represented by sets of vertices of G
        edge_data_type = Set{keytype(PT1)}, # hyperarcs represented by sets of keys of A
        graph_data = "right-resolving representation"
    )

    R[Set((nv(G),))] = nothing  # add first vertex of G as hypernode
    to_prune = [1]

    while !isempty(to_prune)

        to_search = subshift_search(G, R)    
        while !isempty(to_search)
            subshift_extend!(P, G, R, to_search) 
        end

        to_prune = [label_for(R, i) for i in Graphs.vertices(R) if isempty(inneighbors(R, i))]
        #@debug "prune queue" to_prune
        for H in to_prune
            delete!(R, H)
        end
        
        #@debug "graph so far" R
    end

    return R
end

function subshift_search(G::BoxGraph, R::MetaGraph)
    to_search = Set{Int}[]
    for i in Graphs.vertices(R)
        isempty(outneighbors(R, i)) || continue
        H = label_for(R, i)
        all( j -> isempty(outneighbors(G, j)), H ) && continue
        push!(to_search, H)
    end
    #@debug "search queue" to_search
    return to_search
end

function subshift_extend!(P::PT1, G::BoxGraph, R::MetaGraph, to_search) where {PT1<:AbstractBoxPartition}
    Q = G.gstar.domain.partition
    H = pop!(to_search)
    seen_labels = Dict{keytype(PT1),typeof(H)}()
            
    for i in H
        u = index_to_key(G, i)
        c, _ = key_to_box(Q, u)
        lab = point_to_key(P, c)
        #H♯ = get(seen_labels, lab, Set{Int}())
        H♯ = haskey(seen_labels, lab) ? seen_labels[lab] : Set{Int}()
        seen_labels[lab] = union!(H♯, outneighbors(G, i))
    end

    for (lab, H♯) in seen_labels
        isempty(H♯) && continue
        if !haskey(R, H♯)
            R[H♯] = nothing
            push!(to_search, H♯)
        end
        add_lab = haskey(R, H, H♯) ? R[H, H♯] : Set{keytype(PT1)}()
        R[H, H♯] = push!(add_lab, lab)
    end
end

#=

function labelled_graph(P::AbstractBoxPartition, G::BoxGraph)
    Q = G.gstar.domain.partition
    
    vertices_data = map(Graphs.vertices(G)) do i
        u = index_to_key(G, i)
        u => nothing
    end

    edges_data = map(Graphs.edges(G)) do e
        i, j = e.src, e.dst
        u, v = index_to_key(G, i), index_to_key(G, j)
        c, _ = key_to_box(Q, u)
        label = point_to_key(P, c)
        (u, v) => label
    end
    
    return MetaGraph(G, vertices_data, edges_data, "topological Markov chain")
end

function right_resolving_representation(G::MetaGraph{<:Any,<:Any,Q2,<:Any,Q1}) where {Q1,Q2}
    R = MetaGraph(
        DiGraph(); 
        label_type = Set{Q2}, # hypernodes represented by sets of keys of B
        edge_data_type = Set{Q1}, # hyperarcs represented by sets of keys of A
        graph_data = "right-resolving representation"
    )

    ind_1 = Set([ label_for(G, (first ∘ Graphs.vertices)(G)) ])
    R[ind_1] = nothing

    to_prune = [ (first ∘ Graphs.vertices)(R) ]
    labels_A = Set(map(e -> G[e...], edge_labels(G)))

    while !isempty(to_prune)
        @show to_search = [i for i in Graphs.vertices(R) if isempty(outneighbors(R, i))]

        while !isempty(to_search)
            i = pop!(to_search)
            H = label_for(R, i)

            for label in labels_A
                H♯ = Set(label_for(G, j) for u in H for j in outneighbors(G, code_for(G, u)) if G[u, label_for(G, j)] == label)
                haskey(R, H♯) || (R[H♯] = nothing)

                if haskey(R, H, H♯)
                    lab = R[H, H♯]
                    label_out = push!(lab, label)
                else
                    label_out = Set{Q1}([label])
                end

                seen = haskey(R, H♯)
                R[H, H♯] = label_out
                seen || push!(to_search, code_for(R, H♯))
            end

        end

        @show to_prune = [i for i in Graphs.vertices(R) if isempty(inneighbors(R, i))]
        rem_vertices!(R, to_prune)
    end

    return R
end

=#

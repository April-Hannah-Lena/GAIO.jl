const SVNT{N,T} = Union{<:NTuple{N,T}, <:StaticVector{N,T}}
const default_box_color = :red # default color for plotting

# we need a small helper function because of 
# how julia dispatches on `union!`
⊔(set1::AbstractSet, set2::AbstractSet) = union!(set1, set2)
⊔(set1::AbstractSet, object) = union!(set1, (object,))
⊔(set1::AbstractSet, ::Nothing) = set1

⊔(d::AbstractDict...) = mergewith!(+, d...)
⊔(d::AbstractDict, p::Pair...) = foreach(q -> d ⊔ q, p)
⊔(d::AbstractDict, ::Nothing) = d
⊔(d::AbstractDict, ::Pair{<:Tuple{Nothing,<:Any},<:Any}) = d

function ⊔(d::AbstractDict, p::Pair)
    k, v = p
    d[k] = haskey(d, k) ? d[k] + v : v
    d
end

"""
    @interruptable @floop for ...
        ...
    end

Modify an `@threads` or `@floop` macro call to include interrupt 
checking. This ensures that if one thread errors (e.g. from a 
KeyboardInterrupt), the other threads are guaranteed to stop at 
the latest after their iteration. 
"""
macro interruptable(expr)

    # Ensure we are in the right situation
    @assert expr.head == Symbol("macrocall") && expr.args[3].head == Symbol("for")
    newexpr = copy(expr)
    forblock = newexpr.args[3]

    # Add an interrupt checker to the FLoop
    forblock.args[2] = quote
        try
            stop[] && break
            $(forblock.args[2])
        catch err
            Base.Threads.atomic_or!(stop, true)
            rethrow(err)
        end
    end

    # in case of interrupt, throw a global error after loop
    newexpr = quote
        stop = Base.Threads.Atomic{Bool}(false)
        $newexpr
        stop[] && @error "Loop errored."
    end

    return esc(newexpr)
end

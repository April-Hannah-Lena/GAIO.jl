mutable struct KoopmanOperator{B,T,S,M,L<:TransferOperator{B,T,S,M},K,F<:BoxFun{B,K,T}} <: AbstractSparseMatrix{T,Int}
    transfer::L
    invariant_μ::F
end

function KoopmanOperator(transfer::TransferOperator{B,T}) where {B,T}
    λs, evs = eigs(transfer, which=:LR)
    best = argmin(x -> abs(x-1), λs)
    λ, invariant_μ = λs[best], evs[best]
    invariant_μ = (x -> convert(T, x)) ∘ real ∘ invariant_μ
    return KoopmanOperator(transfer, invariant_μ)
end

@propagate_inbounds function LinearAlgebra.mul!(y::AbstractVector, g::KoopmanOperator, x::AbstractVector)
    @boundscheck((length(x), length(y)) == size(g.transfer.mat') || throw(DimensionMismatch("$g, $x")))
    zer = zero(eltype(y))
    map!(x -> zer, values(y.vals))
    rows = rowvals(g.transfer.mat)
    vals = nonzeros(g.transfer.mat)
    # iterate over columns with the keys that the columns represent
    for (col_j, j) in enumerate(g.transfer.domain.set)
        for k in nzrange(g.transfer.mat, col_j)
            w = vals[k]
            row_i = rows[k]
            # grab the key that this row represents
            i = index_to_key(g.transfer.codomain, row_i)
            p = g.invariant_μ[j] / g.invariant_μ[i]
            y[col_j] = @muladd y[col_j] + adjoint(w*p) * x[row_i]
        end
    end
    return y
end

@propagate_inbounds function LinearAlgebra.mul!(y::BoxFun, g::KoopmanOperator, x::BoxFun)
    @boundscheck(checkbounds(Bool, g, keys(x.vals)) || throw(BoundsError(g, x)))
    zer = zero(eltype(y))
    map!(x -> zer, values(y.vals))
    rows = rowvals(g.transfer.mat)
    vals = nonzeros(g.transfer.mat)
    # iterate over columns with the keys that the columns represent
    for (col_j, j) in enumerate(g.transfer.domain.set)
        for k in nzrange(g.transfer.mat, col_j)
            w = vals[k]
            row_i = rows[k]
            # grab the key that this row represents
            i = index_to_key(g.transfer.codomain, row_i)
            p = g.invariant_μ[j] / g.invariant_μ[i]
            y[j] = @muladd y[j] + adjoint(w*p) * x[i]
        end
    end
    return y
end



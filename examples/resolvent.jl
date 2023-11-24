using GAIO
using SparseArrays, LinearAlgebra, KrylovKit, ProgressMeter
using DSP
using Plots

# --------------------------------

f( (x,); α=2, β=-1-exp(-α) ) = ( exp(-α*x^2) + β ,)

domain = Box([-0.5], [0.5])
P = BoxPartition(domain, (1024,))
S = cover(P, :)
F = BoxMap(:interval, f, domain)

# ---------------------------------

f(z; α=-1.7) = exp(-2*pi*im/3) * ( (abs(z)^2 + α)*z + conj(z)^2 / 2 )
fr((x, y)) = reim( f(x + y*im) )

c, r = (0, 0), (1.5, 1.5)
domain = Box(c, r)
P = BoxPartition(domain, (128,128))
F = BoxMap(:interval, fr, domain)

S = cover(P, (0,0))
S = unstable_set(F, S)

# ---------------------------------

F♯ = TransferOperator(F, S, S)

M = similar(F♯.mat', ComplexF64)
M .= F♯.mat'

n = size(M, 1)
x0 = rand(ComplexF64, n) .* exp.(2pi*im .* rand(Float64, n))

function res(z; kwargs...)
    vals, rvecs, lvecs, info = svdsolve(M - z*I, x0, 1, :SR; kwargs...)
    return minimum(vals)
end

xs = -1.1:0.05:1.1
R = @showprogress broadcast(xs', xs) do x, y
    res(x+y*im)
end

using Serialization
serialize("resolvents.ser", R)
R = deserialize("resolvents.ser")

# remove basically-0 revolvents
R̄ = R
R̄[R̄ .< 1e-9] .= 1e-9

# smoothing
A = 0.25 * [1 1;
            1 1]

R̄ = conv(R̄, A)[1:end-1, 1:end-1]

ticks = -9:1:0
labs = ["10^$x" for x in ticks]

p = contourf(
    xs, xs, log10.(R̄), 
    levels=9, #clabels=true,
    colorbar_ticks=(ticks,labs)
)

heatmap(xs, xs, log10.(R))

λ, ev, nconv = eigs(F♯, nev=128)

scatter!(real.(λ), imag.(λ), color=:blue)
for r in 1:0.1:0.1
    plot!(map(t -> r .* (cos(t), sin(t)), 0:0.01:2π))
end
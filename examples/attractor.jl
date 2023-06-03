using GAIO

# the Henon map
a, b = 1.4, 0.3
f((x,y)) = (1 - a*x^2 + y, b*x)

center, radius = (0, 0), (3, 3)
P = BoxPartition(Box(center, radius))
F = BoxMap(f, P)
S = cover(P, :)
A1 = relative_attractor(F, S, steps = 4)
A2 = relative_attractor(F, S, steps = 22)
P = A1.partition
T = TransferOperator(F, A2, A2)
G = BoxGraph(T)
using Graphs, MetaGraphsNext
using GAIO: point_to_key, key_to_box, index_to_key, key_to_index, ⊔
ENV["JULIA_DEBUG"] = "all"

using Plots: plot
#using WGLMakie: plot    # same result, just interactive

plot(A)

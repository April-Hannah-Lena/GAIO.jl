using GAIO

const a, b, c, d, e, g, h = 0.2, -0.01, 1.0, -0.4, -1.0, -1.0, -1.0
v((x,y,z)) = @. (a,d,e)*(x,y,z) + (0,b*x,0) + (c,h,g)*(y,z,x)*(z,x,y)

f(x) = rk4_flow_map(v, x)

cen, rad = (0,0,0), (5,5,5)
dom = Box(cen, rad)
P = BoxPartition(dom, (128,128,128))

F = BoxMap(:adaptive, f, dom)

S = cover(P, cen)
S = S ∪ nbhd(S)
W = unstable_set(F, S)

T = TransferOperator(F, W, W)
(λ, ev) = eigs(T; nev=12)
μ = (x -> sign(x) * log(abs(x))) ∘ real ∘ ev[2]

# --- choose either Plots or Makie ---

using Plots: plot
# Plot some 2D projections
p1 = plot(μ, projection = x->x[1:2])
p2 = plot(μ, projection = x->x[2:3])
plot(p1, p2, size = (1200,600))

# ------------------------------------

using GLMakie
# Plot an interactive 3D heatmap
fig, ax, ms = plot(μ, colormap=(:buda,0.5))
Colorbar(fig[1,2], ms)
fig

τ = 0
B1 = BoxSet(P, Set(key for key in keys(μ) if μ[key] ≤ τ))
B2 = BoxSet(P, Set(key for key in keys(μ) if μ[key] > τ))

fig = Figure();
ax1 = Axis3(fig[1,1])
ms1 = plot!(ax1, B1, color=(:blue, 0.4))
ax2 = Axis3(fig[1,2])
ms2 = plot!(ax2, B2, color=(:red, 0.4))
fig

# ------------------------------------


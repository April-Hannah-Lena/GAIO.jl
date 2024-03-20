using GAIO, StaticArrays, LaTeXStrings, Serialization

const a, b, c, d, e, g, h = 0.2, -0.01, 1.0, -0.4, -1.0, -1.0, -1.0
v((x,y,z)) = @. (a,d,e)*(x,y,z) + 
                    (0,b*x,0) + 
                    (c,h,g)*(y,z,x)*(z,x,y)

f(x) = rk4_flow_map(v, x)

cen, rad = (0,0,0), (5,5,5)
dom = Box(cen, rad)
P = BoxPartition(dom, (2,2,2))

F = BoxMap(:adaptive, f, dom)

S = cover(P, :)
A = maximal_backward_invariant_set(F, S, steps=21)

using CairoMakie
begin
    fig = Figure(backgroundcolor=:transparent);
    ax1 = Axis3(fig[1,1], aspect=(1,1,1), azimuth=5π/4, elevation=π/8, backgroundcolor=:transparent);
    ax2 = Axis3(fig[1,2], aspect=(1,1,1), azimuth=π/16, elevation=π/16, backgroundcolor=:transparent);
    ax3 = Axis3(fig[2,1], aspect=(1,1,1), azimuth=7π/4, elevation=3π/8, backgroundcolor=:transparent);
    ax4 = Axis3(fig[2,2], aspect=(1,1,1), azimuth=7π/16, elevation=π/16, backgroundcolor=:transparent);
    ms1 = plot!(ax1, A, color=(:red, 0.4));
    ms2 = plot!(ax2, A, color=(:red, 0.4));
    ms3 = plot!(ax3, A, color=(:red, 0.4));
    ms4 = plot!(ax4, A, color=(:red, 0.4));
end
save("attractor.png", fig, px_per_unit=4)

P2 = BoxPartition(dom, (8,8,8))
A2 = cover(P2, A)

begin
    fig = Figure(backgroundcolor=:transparent);
    ax1 = Axis3(fig[1,1], aspect=(1,1,1), azimuth=5π/4, elevation=π/8, backgroundcolor=:transparent);
    ax2 = Axis3(fig[1,2], aspect=(1,1,1), azimuth=π/16, elevation=π/16, backgroundcolor=:transparent);
    ax3 = Axis3(fig[2,1], aspect=(1,1,1), azimuth=7π/4, elevation=3π/8, backgroundcolor=:transparent);
    ax4 = Axis3(fig[2,2], aspect=(1,1,1), azimuth=7π/16, elevation=π/16, backgroundcolor=:transparent);
    for box in A2
        c2, a2 = box
        x = SVector{3,Float32}[c2]
        for _ in 2:600
            push!(x, rk4_flow_map(v, x[end], 0.01, 1))
        end
        indices = [101:600;]
        x = x[indices]
        ms1 = lines!(ax1, x, color=[indices;], colormap=(:blues, 0.25));
        ms2 = lines!(ax2, x, color=[indices;], colormap=(:blues, 0.25));
        ms3 = lines!(ax3, x, color=[indices;], colormap=(:blues, 0.25));
        ms4 = lines!(ax4, x, color=[indices;], colormap=(:blues, 0.25));
    end
end
save("trajectories.png", fig, px_per_unit=4)

P = BoxPartition(dom, (128,128,128))
S = cover(P, cen)
W = unstable_set(F, S)

T = TransferOperator(F, W, W)
(λ, ev) = eigs(T; nev=2)
μ = log10 ∘ abs ∘ ev[1]

begin
    fig = Figure(backgroundcolor=:transparent);
    ax1 = Axis3(fig[1,1], aspect=(1,1,1), azimuth=5π/4, elevation=π/8, backgroundcolor=:transparent);
    ax2 = Axis3(fig[1,2], aspect=(1,1,1), azimuth=π/16, elevation=π/16, backgroundcolor=:transparent);
    ax3 = Axis3(fig[2,1], aspect=(1,1,1), azimuth=7π/4, elevation=3π/8, backgroundcolor=:transparent);
    ax4 = Axis3(fig[2,2], aspect=(1,1,1), azimuth=7π/16, elevation=π/16, backgroundcolor=:transparent);
    ms1 = plot!(ax1, μ, colormap=(:jet, 0.4))
    ms2 = plot!(ax2, μ, colormap=(:jet, 0.4))
    ms3 = plot!(ax3, μ, colormap=(:jet, 0.4))
    ms4 = plot!(ax4, μ, colormap=(:jet, 0.4))
    Colorbar(fig[1:2,3], ms1, ticks=([-5,-10,-15,-20], [L"10^{-5}", L"10^{-10}", L"10^{-15}", L"10^{-20}"]))
end
save("inv_measure.png", fig, px_per_unit=4)


import CairoMakie.Makie: Symlog10, ReversibleScale
symlog(γ) = ReversibleScale(
    x -> sign(x) * (log10(abs(x) + γ) - log10(γ)),
    x -> sign(x) * γ * (10^(abs(x)) - 1)
)
scale = symlog(1e-25)
μ = scale ∘ real ∘ ev[2]
plot(μ, colormap=(:jet, 0.4))

begin
    fig = Figure(backgroundcolor=:transparent);
    ax1 = Axis3(fig[1,1], aspect=(1,1,1), azimuth=5π/4, elevation=π/8, backgroundcolor=:transparent);
    ax2 = Axis3(fig[1,2], aspect=(1,1,1), azimuth=π/16, elevation=π/16, backgroundcolor=:transparent);
    ax3 = Axis3(fig[2,1], aspect=(1,1,1), azimuth=7π/4, elevation=3π/8, backgroundcolor=:transparent);
    ax4 = Axis3(fig[2,2], aspect=(1,1,1), azimuth=7π/16, elevation=π/16, backgroundcolor=:transparent);
    ms1 = plot!(ax1, μ, colormap=(:buda, 0.4))
    ms2 = plot!(ax2, μ, colormap=(:buda, 0.4))
    ms3 = plot!(ax3, μ, colormap=(:buda, 0.4))
    ms4 = plot!(ax4, μ, colormap=(:buda, 0.4))
    Colorbar(fig[1:2,3], ms1, ticks=([-16,-6,0,6,16], [L"-10^{-10}", L"-10^{-20}", L"0", L"10^{-20}", L"10^{-10}"]))
end
save("almost_inv.png", fig, px_per_unit=4)

tol, maxiter = eps()^0.25, 1000
λ, ev, nconv = eigs(T; which=:LM, nev=800, tol=tol, maxiter=maxiter)
scatter(λ)

λ2, ev2, nconv = eigs(T; which=:SR, nev=300, tol=tol, maxiter=maxiter)
scatter(λ2)

#serialize("lambda.ser", [λ; λ2])
λ = deserialize("./paper/lambda.ser")
begin
    fig = Figure(backgroundcolor=:transparent);
    ax = Axis(fig[1,1], xlabel="Re", ylabel="Im", backgroundcolor=:transparent)
    ms = scatter!(ax, λ)
end
save("spectrum.pdf", fig, pt_per_unit=4)

println("done")

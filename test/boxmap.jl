using GAIO
using StaticArrays
using Test

@testset "exported functionality" begin
    f(x) = x .^ 2
    test_points = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
    center = (0.0, 0.0)
    radius = (1.0, 1.0)
    domain = Box(center, radius)
    g = PointDiscretizedMap(f, domain, test_points)
    @testset "basics" begin
        @test typeof(g) <: SampledBoxMap
        partition = BoxPartition(domain, (32,32))
        p1 = (0.0, 0.0)
        p2 = (0.5, 0.0)
        p3 = (0.0, -0.5)
        boxset = partition[(p1, p2, p3)]
        # re-implement a straight forward box map to have a ground truth
        # terrible implementation in any real scenario
        mapped_points = []
        for box in boxset
            push!(mapped_points, f(box.center .- box.radius))
            push!(mapped_points, f(box.center .+ box.radius))
            x = f((box.center[1] - box.radius[1], box.center[2] + box.radius[2]))
            push!(mapped_points, x)
            y = f((box.center[1] + box.radius[1], box.center[2] - box.radius[2]))
            push!(mapped_points, y)
        end
        image = partition[mapped_points]
        full_boxset = partition[:]
        mapped1 = g(boxset)

        @test !isempty(boxset) && !isempty(image)
        # easiest way right now to check for equality
        @test length(union(image, mapped1)) == length(intersect(image, mapped1))
    end
    @testset "points in boxmaps" begin
        x = (-2.0, 3.0)
        y = (4.0, 1.0)
        @test_throws MethodError g(x)
        @test_throws MethodError g(y)
    end
end

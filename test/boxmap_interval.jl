using GAIO
using StaticArrays
using IntervalArithmetic
using Test

@testset "exported functionality" begin
    f(x) = x .^ 2
    center = SVector(0.0, 0.0)
    radius = SVector(1.0, 1.0)
    domain = Box(center, radius)
    g = BoxMap(:interval, f, domain, n_subintervals=(1,1))
    @testset "basics" begin
        @test typeof(g) <: IntervalBoxMap
        partition = BoxPartition(domain, (32,32))
        p1 = SVector(0.0, 0.0)
        p2 = SVector(0.5, 0.0)
        p3 = SVector(0.0, -0.5)
        boxset = cover(partition, (p1, p2, p3))
        mapped1 = g(boxset)

        boxarr = collect(IntervalBox(c .± r ...) for (c,r) in boxset)
        image_arr = collect(Box(f(int)) for int in boxarr)
        image_set = cover(partition, image_arr)

        @test image_set == mapped1
    end
end

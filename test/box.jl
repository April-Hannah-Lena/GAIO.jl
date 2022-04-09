using GAIO
using StaticArrays
using Test

@testset "exported functionality" begin
    @testset "basics" begin
        center = (0.0, 0.1)
        radius = (10.0, 10.0)
        box = Box(center, radius)
        @test box.center == center
        @test box.radius == radius
    end
    @testset "types" begin
        center = (0., 0., 1)
        radius = (1.0, 0.1, 1.0)
        box = Box(center, radius)
        @test typeof(box.center) <: typeof(box.radius)
        @test typeof(box.radius) <: typeof(box.center)
        @test !(typeof(box.center) <: typeof(center))
    end
    @testset "containment" begin
        center = (0.0, 0.0, 0.0)
        radius = (1.0, 1.0, 1.0)
        box = Box(center, radius)
        inside = (0.5, 0.5, 0.5)
        left = (-1.0, -1.0, -1.0)
        right = (1.0, 1.0, 1.0)
        on_boundary_left = (0.0, 0.0, -1.0)
        on_boundary_right = (0.0, 1.0, 0.0)
        outside_left = (0.0, 0.0, -2.0)
        outside_right = (0.0, 2.0, 0.0)
        @test inside ∈ box
        @test box.center ∈ box
        #boxes are half open to the right
        @test left ∈ box
        @test right ∉ box
        @test on_boundary_left ∈ box
        @test on_boundary_right ∉ box
        @test outside_left ∉ box
        @test outside_right ∉ box
    end
    @testset "non matching dimensions" begin
        center = (0.0, 0.0, 0.0)
        radius = (1.0, 1.0)
        @test_throws Exception Box(center, radius)
    end
    @testset "negative radii" begin
        center = (0.0, 0.0)
        radius = (1.0, -1.0)
        @test_throws Exception Box(center, radius)
    end
end
@testset "internal functionality" begin
    box = Box((0.0, 0.0), (1.0, 1.0))
    @testset "integer point in box" begin
        point_int_outside = (2, 2)
        point_int_inside = (0, 0)
        @test point_int_inside ∈ box
        @test point_int_outside ∉ box
    end
    @test_throws DimensionMismatch (0.0, 0.0, 0.0) ∈ box
end

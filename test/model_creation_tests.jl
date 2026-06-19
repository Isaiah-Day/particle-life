using ParticleLife, Test

@testset "Particle Bounds Check" begin
    for i in 1:10
        m = create_model(seed=i)
        download_positions!(m)
        @test maximum(cat(m.cpu_px, m.cpu_py; dims=1)) <= 1
        @test minimum(cat(m.cpu_py, m.cpu_py; dims=1)) >= 0

        for _ in 1:100
            model_step!(m)
        end
        download_positions!(m)
        @test maximum(cat(m.cpu_px, m.cpu_py; dims=1)) <= 1
        @test minimum(cat(m.cpu_py, m.cpu_py; dims=1)) >= 0
    end
end

@testset "Particle Zero Attraction Check" begin
    m0 = create_model(attraction_matrix=zeros(Float32, 6, 6))
    m1 = deepcopy(m0)

    for _ in 1:100
        model_step!(m1)
    end

    @test m0.cpu_px == m1.cpu_px
    @test m0.cpu_py == m1.cpu_py
end
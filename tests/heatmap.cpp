#include "common.hpp"

#include <catch2/catch.hpp>
#include <fstream>
#include <llama/llama.hpp>

TEST_CASE("Heatmap.nbody")
{
    constexpr auto N = 100;
    auto run = [&](const std::string& name, auto mapping)
    {
        auto particles = llama::allocView(llama::mapping::Heatmap{mapping});

        constexpr float TIMESTEP = 0.0001f;
        constexpr float EPS2 = 0.01f;
        for(std::size_t i = 0; i < N; i++)
        {
            llama::One<ParticleHeatmap> pi = particles(i);
            for(std::size_t j = 0; j < N; ++j)
            {
                auto pj = particles(j);
                auto dist = pi(tag::Pos{}) - pj(tag::Pos{});
                dist *= dist;
                const float distSqr = EPS2 + dist(tag::X{}) + dist(tag::Y{}) + dist(tag::Z{});
                const float distSixth = distSqr * distSqr * distSqr;
                const float invDistCube = 1.0f / std::sqrt(distSixth);
                const float sts = pj(tag::Mass{}) * invDistCube * TIMESTEP;
                pi(tag::Vel{}) += dist * sts;
            }
            particles(i) = pi;
        }
        for(std::size_t i = 0; i < N; i++)
            particles(i)(tag::Pos{}) += particles(i)(tag::Vel{}) * TIMESTEP;

        std::ofstream{"Heatmap." + name + ".sh"} << particles.mapping().toGnuplotScript();
    };

    using ArrayDims = llama::ArrayExtents<N>;
    auto arrayDims = ArrayDims{};
    run("AlignedAoS", llama::mapping::AlignedAoS<ArrayDims, ParticleHeatmap>{arrayDims});
    run("SingleBlobSoA", llama::mapping::SingleBlobSoA<ArrayDims, ParticleHeatmap>{arrayDims});
}

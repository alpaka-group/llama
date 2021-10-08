#include "common.hpp"

#include <fstream>

namespace
{
    constexpr auto N = 100;

    template<typename View>
    auto updateAndMove(View& particles)
    {
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
    }
} // namespace

TEST_CASE("Heatmap.nbody")
{
    auto run = [&](const std::string& name, auto mapping)
    {
        auto particles = llama::allocView(llama::mapping::Heatmap{mapping});
        updateAndMove(particles);
        std::ofstream{"Heatmap." + name + ".sh"} << particles.mapping().toGnuplotScript();
    };
    run("AlignedAoS", llama::mapping::AlignedAoS<llama::ArrayExtents<N>, ParticleHeatmap>{});
    run("SingleBlobSoA", llama::mapping::SingleBlobSoA<llama::ArrayExtents<N>, ParticleHeatmap>{});
}

TEST_CASE("Trace.nbody")
{
    auto run = [&](auto mapping)
    {
        auto particles = llama::allocView(llama::mapping::Trace{mapping, false});
        updateAndMove(particles);
        auto& hits = particles.mapping().fieldHits;
        CHECK(hits.at("NoName.Pos.X") == 10400);
        CHECK(hits.at("NoName.Pos.Y") == 10400);
        CHECK(hits.at("NoName.Pos.Z") == 10400);
        CHECK(hits.at("NoName.Vel.X") == 400);
        CHECK(hits.at("NoName.Vel.Y") == 400);
        CHECK(hits.at("NoName.Vel.Z") == 400);
        CHECK(hits.at("NoName.Mass") == 10300);
    };
    run(llama::mapping::AlignedAoS<llama::ArrayExtents<N>, ParticleHeatmap>{});
    run(llama::mapping::SingleBlobSoA<llama::ArrayExtents<N>, ParticleHeatmap>{});
}

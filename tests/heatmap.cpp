#include <catch2/catch.hpp>
#include <fstream>
#include <llama/llama.hpp>

namespace
{
    // clang-format off
    namespace tag
    {
        struct Pos{};
        struct Vel{};
        struct X{};
        struct Y{};
        struct Z{};
        struct Mass{};
    } // namespace tag

    using Particle = llama::DS<
        llama::DE<tag::Pos, llama::DS<
            llama::DE<tag::X, float>,
            llama::DE<tag::Y, float>,
            llama::DE<tag::Z, float>
        >>,
        llama::DE<tag::Vel, llama::DS<
            llama::DE<tag::X, float>,
            llama::DE<tag::Y, float>,
            llama::DE<tag::Z, float>
        >>,
        llama::DE<tag::Mass, float>
    >;
    // clang-format on
} // namespace

TEST_CASE("Heatmap.3body")
{
    constexpr auto N = 100;
    auto run = [&](const std::string& name, auto mapping)
    {
        auto particles = llama::allocView(llama::mapping::Heatmap{mapping});

        for (std::size_t i = 0; i < N; i++)
            particles(i) = 0;

        constexpr float TIMESTEP = 0.0001f;
        constexpr float EPS2 = 0.01f;
        for (std::size_t i = 0; i < N; i++)
            for (std::size_t j = 0; j < N; ++j)
            {
                auto pi = particles(i);
                auto pj = particles(j);
                auto dist = pi(tag::Pos{}) - pj(tag::Pos{});
                dist *= dist;
                const float distSqr = EPS2 + dist(tag::X{}) + dist(tag::Y{}) + dist(tag::Z{});
                const float distSixth = distSqr * distSqr * distSqr;
                const float invDistCube = 1.0f / std::sqrt(distSixth);
                const float sts = pj(tag::Mass{}) * invDistCube * TIMESTEP;
                pi(tag::Vel{}) += dist * sts;
            }
        for (std::size_t i = 0; i < N; i++)
            particles(i)(tag::Pos{}) += particles(i)(tag::Vel{}) * TIMESTEP;

        std::ofstream{"Heatmap." + name + ".dat"} << particles.mapping.toGnuplotDatFile();
    };

    using ArrayDomain = llama::ArrayDomain<1>;
    auto arrayDomain = ArrayDomain{N};
    run("AlignedAoS", llama::mapping::AlignedAoS<ArrayDomain, Particle>{arrayDomain});
    run("SingleBlobSoA", llama::mapping::SingleBlobSoA<ArrayDomain, Particle>{arrayDomain});
}

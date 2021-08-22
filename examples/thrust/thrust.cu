//#define THRUST_DEBUG_SYNC
#include "../common/Stopwatch.hpp"
#include "../common/hostname.hpp"

#include <algorithm>
#include <boost/preprocessor/stringize.hpp>
#include <execution>
#include <fmt/format.h>
#include <fstream>
#include <iomanip>
#include <llama/llama.hpp>
#include <numeric>
#include <omp.h>
#include <random>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
//#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/shuffle.h>
#include <thrust/tabulate.h>
//#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>

constexpr auto REPETITIONS = 5;
constexpr auto N = std::size_t{1024} * 1024;
constexpr auto shuffle_frac = 20; // 1/x fraction of the data to shuffle
constexpr auto usePSTL = false;
// const auto exec = std::execution::par_unseq;
const auto exec = std::execution::seq;

constexpr auto thrustUsesCuda = THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA;
static_assert(!(thrustUsesCuda && usePSTL), "Cannot use thrust/CUDA and PSTL together");

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

using Vec3 = llama::Record<
    llama::Field<tag::X, float>,
    llama::Field<tag::Y, float>,
    llama::Field<tag::Z, float>
>;

using Particle = llama::Record<
    llama::Field<tag::Pos, Vec3>,
    llama::Field<tag::Vel, Vec3>,
    llama::Field<tag::Mass, float>
>;

namespace tag
{
    struct rng_state{};
    struct index{};
    struct mother_index{};
    struct eventId{};

    struct energy{};
    struct energy_loss{};
    struct numIALeft{};
    struct interaction_length{};
    struct total_length{};

    struct pos{};
    struct dir{};
    struct current_state{};
    struct next_state{};
    
    struct status{};

    struct number_of_secondaries{};

    struct num_step{};

    struct current_process{};
    struct pdg{};
} // namespace tag

using Vec3D = llama::Record<
    llama::Field<tag::X, double>,
    llama::Field<tag::Y, double>,
    llama::Field<tag::Z, double>
>;

struct RngState
{
    std::byte state[240];

    LLAMA_FN_HOST_ACC_INLINE auto operator=(double d) -> RngState&
    {
        for (int i = 0; i < 240; i++)
            state[i] = static_cast<std::byte>(d);
        return *this;
    }
};

// inspired by: https://github.com/apt-sim/AdePT/blob/master/tracking/inc/track.h
// packed size: 621 B
// padded size: 632 B
using Track = llama::Record<
    llama::Field<tag::rng_state, RngState>,
    llama::Field<tag::index, unsigned int>,
    llama::Field<tag::mother_index, unsigned int>,
    llama::Field<tag::eventId, unsigned int>,

    llama::Field<tag::energy, double>,
    llama::Field<tag::energy_loss, double>,
    llama::Field<tag::numIALeft, double[3]>,
    llama::Field<tag::interaction_length, double>,
    llama::Field<tag::total_length, double>,

    llama::Field<tag::pos, double[3]>,
    llama::Field<tag::dir, double[3]>,

    llama::Field<tag::current_state, double[4*4]>, // matrix
    llama::Field<tag::next_state, double[4*4]>,    // matrix

    llama::Field<tag::status, bool>,

    llama::Field<tag::number_of_secondaries, int>,
    llama::Field<tag::num_step, std::uint16_t>,

    llama::Field<tag::current_process, char>,
    llama::Field<tag::pdg, char>
>;
// clang-format on

#define USE_TRACK

#ifdef USE_TRACK
using RecordDim = Track;
#else
using RecordDim = Particle;
#endif

// struct Struct
//{
//    float posx;
//    float posy;
//    float posz;
//    float velx;
//    float vely;
//    float velz;
//    float mass;
//};
//
// LLAMA_FN_HOST_ACC_INLINE auto operator<(Struct a, Struct b) -> bool
//{
//    return a.mass < b.mass;
//}

// auto less = [] LLAMA_FN_HOST_ACC_INLINE(auto&& a, auto&& b)
//{
//    // printf("a < b: %p %p\n", &a, &b);
//    return a < b;
//};
// auto less = [] LLAMA_FN_HOST_ACC_INLINE (auto&& a, auto&& b)
//{
//    auto apos = a(tag::Pos{});
//    auto bpos = b(tag::Pos{});
//    return apos(tag::X{}) + apos(tag::Y{}) + apos(tag::Z{}) < bpos(tag::X{}) + bpos(tag::Y{}) + bpos(tag::Z{});
//};

// struct Less
//{
//    template<typename A, typename B>
//    LLAMA_FN_HOST_ACC_INLINE auto operator()(A&& a, B&& b) -> bool
//    {
//        auto posA = a(tag::Pos{});
//        auto posB = b(tag::Pos{});
//        // return a(tag::Mass{}) < b(tag::Mass{});
//        return posA(tag::X{}) + posA(tag::Y{}) + posA(tag::Z{}) < posB(tag::X{}) + posB(tag::Y{}) + posB(tag::Z{});
//    }
//};

// due to abominal HACKs in LLAMA to get thrust working, all writing accesses on llama::One must be terminal, because
// the llama::One copies its storage on each non-terminal access
struct NormalizeVel
{
    template<typename VirtualRecord>
    LLAMA_FN_HOST_ACC_INLINE void operator()(VirtualRecord&& p)
    {
#ifdef USE_TRACK
        double& x = p(tag::dir{}, llama::RecordCoord<0>{});
        double& y = p(tag::dir{}, llama::RecordCoord<1>{});
        double& z = p(tag::dir{}, llama::RecordCoord<2>{});
        const auto invNorm = 1.0 / std::sqrt(x * x + y * y + z * z);
        x *= invNorm;
        y *= invNorm;
        z *= invNorm;
#else
        float& x = p(tag::Vel{}, tag::X{});
        float& y = p(tag::Vel{}, tag::Y{});
        float& z = p(tag::Vel{}, tag::Z{});
        const auto invNorm = 1.0f / std::sqrtf(x * x + y * y + z * z);
        x *= invNorm;
        y *= invNorm;
        z *= invNorm;
        return p;
        // float x = p(tag::Vel{}, tag::X{});
        // float y = p(tag::Vel{}, tag::Y{});
        // float z = p(tag::Vel{}, tag::Z{});
        // p(tag::Vel{}, tag::X{}) = 0.36f * x + 0.48f * y + -0.80f * z;
        // p(tag::Vel{}, tag::Y{}) = -0.80f * x + 0.60f * y + 0.00f * z;
        // p(tag::Vel{}, tag::Z{}) = 0.48f * x + 0.64f * y + 0.60f * z;
#endif
    }
};

struct Predicate
{
    // LLAMA_FN_HOST_ACC_INLINE int c = 0;

    //~Predicate()
    //{
    //    printf("total %d/%d\n", c, (int)N);
    //}

    template<typename VirtualPoint>
    LLAMA_FN_HOST_ACC_INLINE auto operator()(VirtualPoint&& p) -> bool
    {
#ifdef USE_TRACK
        const auto r = (p(tag::dir{}, llama::RecordCoord<0>{}) + p(tag::dir{}, llama::RecordCoord<1>{})
                        + p(tag::dir{}, llama::RecordCoord<2>{}))
            < -0.6;
        return r;
#else
        // half the particles, 5 taken, 5 not
        // const auto r = p(tag::Vel{}, tag::X{}) < 0;

        // quarter of the particles, 5 taken, 15 not
        // const auto r = p(tag::Vel{}, tag::X{}) < 0 && p(tag::Vel{}, tag::Y{}) < 0;

        // half the particles, 3 taken, 3 not, 7 taken, t not
        // const auto r = (p(tag::Vel{}, tag::X{}) + p(tag::Vel{}, tag::Y{})) < 0;

        // half the particles, 7 taken, 2 not, 4 taken, 6 not, 6 taken, 4 not, 2 taken, 8 not
        // const auto r = (p(tag::Vel{}, tag::X{}) + p(tag::Vel{}, tag::Y{}) + p(tag::Vel{}, tag::Z{})) < 0;

        // 388/1024 of the particles
        const auto r = (p(tag::Vel{}, tag::X{}) + p(tag::Vel{}, tag::Y{}) + p(tag::Vel{}, tag::Z{})) < -0.6;

        // c += r;
        // printf("predicate %i\n", (int) r);
        return r;
#endif
    }
};

struct InitOne
{
    LLAMA_FN_HOST_ACC_INLINE auto operator()(int i)
    {
        const auto fx = static_cast<float>(i % 10) / 9.0f - 0.5f;
        const auto fy = static_cast<float>(i % 20) / 19.0f - 0.5f;
        const auto fz = static_cast<float>(i % 40) / 39.0f - 0.5f;
#ifdef USE_TRACK
        llama::One<RecordDim> p;
        p = static_cast<double>(fx + fy + fz);
        p(tag::dir{}, llama::RecordCoord<0>{}) = fx;
        p(tag::dir{}, llama::RecordCoord<1>{}) = fy;
        p(tag::dir{}, llama::RecordCoord<2>{}) = fz;
        return p;
#else
        llama::One<RecordDim> p;
        p(tag::Vel{}, tag::X{}) = fx;
        p(tag::Vel{}, tag::Y{}) = fy;
        p(tag::Vel{}, tag::Z{}) = fz;
        p(tag::Pos{}, tag::X{}) = fx;
        p(tag::Pos{}, tag::Y{}) = fy;
        p(tag::Pos{}, tag::Z{}) = fz;
        p(tag::Mass{}) = fx + fy + fz;
        return p;
#endif
    }
};

struct GetMass
{
    template<typename VirtualPoint>
    LLAMA_FN_HOST_ACC_INLINE auto operator()(VirtualPoint&& p)
    {
#ifdef USE_TRACK
        return p(tag::energy{});
#else
        return p(tag::Mass{});
#endif
    }
};

void checkError(cudaError_t code)
{
    if(code != cudaSuccess)
        throw std::runtime_error(std::string{"CUDA Error: "} + cudaGetErrorString(code));
}

void syncWithCuda()
{
    if constexpr(thrustUsesCuda)
        checkError(cudaDeviceSynchronize());
}

volatile std::uint32_t sink;
volatile void* sinkp;

auto thrustDeviceAlloc = [](auto alignment, std::size_t size)
{
    auto* p = thrust::device_malloc<std::byte>(size).get();
    assert(reinterpret_cast<std::intptr_t>(p) & (alignment - 1) == 0); // assert device_malloc sufficiently aligns
    return p;
};

template<int Mapping>
void run(std::ostream& plotFile)
{
    auto mappingName = []() -> std::string
    {
        if(Mapping == 0)
            return "AoS";
        if(Mapping == 1)
            return "SoA";
        if(Mapping == 2)
            return "SoA MB";
        if(Mapping == 3)
            return "AoSoA8";
        if(Mapping == 4)
            return "AoSoA16";
        if(Mapping == 5)
            return "AoSoA32";
        if(Mapping == 6)
            return "Split AoS";
        std::abort();
    }();
    auto mapping = [&]
    {
        auto extents = llama::ArrayExtents<std::size_t, llama::dyn>{N};
        if constexpr(Mapping == 0)
            return llama::mapping::AoS{extents, RecordDim{}};
        if constexpr(Mapping == 1)
            return llama::mapping::SoA{extents, RecordDim{}};
        if constexpr(Mapping == 2)
            return llama::mapping::SoA<decltype(extents), RecordDim, true>{extents};
        if constexpr(Mapping == 3)
            return llama::mapping::AoSoA<decltype(extents), RecordDim, 8>{extents};
        if constexpr(Mapping == 4)
            return llama::mapping::AoSoA<decltype(extents), RecordDim, 16>{extents};
        if constexpr(Mapping == 5)
            return llama::mapping::AoSoA<decltype(extents), RecordDim, 32>{extents};
        if constexpr(Mapping == 6)
            return llama::mapping::Split<
                decltype(extents),
                RecordDim,
#ifdef USE_TRACK
                llama::RecordCoord<10>, // dir
#else
                llama::RecordCoord<1>, // Vel
#endif
                llama::mapping::BindAoS<>::fn,
                llama::mapping::BindAoS<>::fn,
                true>{extents};
    }();

    std::cout << mappingName << '\n';

    auto view = llama::allocView(mapping, thrustDeviceAlloc);

    // touch memory once before running benchmarks
    thrust::fill(thrust::device, view.begin(), view.end(), 0);
    syncWithCuda();

    //#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    // thrust::counting_iterator<std::size_t> ci(0);
    // thrust::transform(
    //    thrust::device,
    //    ci,
    //    ci + N,
    //    view.begin(),
    //    [] LLAMA_FN_HOST_ACC_INLINE(std::size_t i)
    //    {
    //        thrust::default_random_engine e;
    //        e.discard(i);
    //        thrust::uniform_real_distribution d{-1.0f, 1.0f};
    //        return d(e);
    //    });
    //#else
    //    {
    //        std::default_random_engine e;
    //        std::uniform_real_distribution d{-1.0f, 1.0f};
    //        for(auto vr : view)
    //            vr = d(e);
    //    }
    //#endif

    double tabulateTotal = 0;
    // double shuffleTotal = 0;
    double forEachTotal = 0;
    double transformTotal = 0;
    double transformScanTotal = 0;
    double transformReduceTotal = 0;
    double copyTotal = 0;
    double copyIfTotal = 0;
    double removeIfTotal = 0;
    // double sortTotal = 0;
    for(auto i = 0; i < REPETITIONS; i++)
    {
        {
            Stopwatch stopwatch;
            if constexpr(usePSTL)
            {
                std::transform(
                    exec,
                    thrust::make_counting_iterator(std::size_t{0}),
                    thrust::make_counting_iterator(N),
                    view.begin(),
                    InitOne{});
            }
            else
            {
                thrust::tabulate(thrust::device, view.begin(), view.end(), InitOne{});
                syncWithCuda();
            }
            tabulateTotal += stopwatch.printAndReset("tabulate", '\t');
        }

        //{
        //    // shuffle less data so runtime stays in the same realm as the other benchmarks
        //    Stopwatch stopwatch;
        //    if constexpr(usePSTL)
        //        std::shuffle(view.begin(), view.begin() + N / shuffle_frac, std::default_random_engine{});
        //    else
        //    {
        //        thrust::shuffle(
        //            thrust::device,
        //            view.begin(),
        //            view.begin() + N / shuffle_frac,
        //            thrust::default_random_engine{});
        //        syncWithCuda();
        //    }
        //    shuffleTotal += stopwatch.printAndReset("shuffle", '\t');
        //}

        {
            Stopwatch stopwatch;
            if constexpr(usePSTL)
                std::for_each(exec, view.begin(), view.end(), NormalizeVel{});
            else
            {
                thrust::for_each(
                    thrust::device,
                    thrust::make_counting_iterator(std::size_t{0}),
                    thrust::make_counting_iterator(N),
                    [view] __host__ __device__(std::size_t i) mutable { NormalizeVel{}(view[i]); });
                syncWithCuda();
            }
            forEachTotal += stopwatch.printAndReset("for_each", '\t');
        }

        using MassType = decltype(GetMass{}(view[0]));
        {
            thrust::device_vector<MassType> dst(N);
            Stopwatch stopwatch;
            if constexpr(usePSTL)
                std::transform(exec, view.begin(), view.end(), dst.begin(), GetMass{});
            else
            {
                thrust::transform(thrust::device, view.begin(), view.end(), dst.begin(), GetMass{});
                syncWithCuda();
            }
            transformTotal += stopwatch.printAndReset("transform", '\t');
            if constexpr(!thrustUsesCuda)
                // make sure the compiler does not optimize away the benchmark
                sink = std::reduce(dst.begin(), dst.end(), MassType{0});
        }

        {
            thrust::device_vector<std::uint32_t> scan_result(N);
            Stopwatch stopwatch;
            if constexpr(usePSTL)
                std::transform_exclusive_scan(
                    exec,
                    view.begin(),
                    view.end(),
                    scan_result.begin(),
                    std::uint32_t{0},
                    std::plus<>{},
                    Predicate{});
            else
            {
                thrust::transform_exclusive_scan(
                    thrust::device,
                    view.begin(),
                    view.end(),
                    scan_result.begin(),
                    Predicate{},
                    std::uint32_t{0},
                    thrust::plus<>{});
                syncWithCuda();
            }
            transformScanTotal += stopwatch.printAndReset("transform_scan", '\t');
            if constexpr(!thrustUsesCuda)
                // make sure the compiler does not optimize away the benchmark
                sink = std::reduce(scan_result.begin(), scan_result.end(), std::uint32_t{0});
        }

        {
            Stopwatch stopwatch;
            if constexpr(usePSTL)
                sink = std::transform_reduce(exec, view.begin(), view.end(), MassType{0}, std::plus<>{}, GetMass{});
            else
            {
                sink = thrust::transform_reduce(
                    thrust::device,
                    view.begin(),
                    view.end(),
                    GetMass{},
                    MassType{0},
                    thrust::plus<>{});
                syncWithCuda();
            }
            transformReduceTotal += stopwatch.printAndReset("transform_reduce", '\t');
        }

        {
            auto dstView = llama::allocView(mapping, thrustDeviceAlloc);
            Stopwatch stopwatch;
            if constexpr(usePSTL)
                std::copy(exec, view.begin(), view.end(), dstView.begin());
            else
            {
                thrust::copy(thrust::device, view.begin(), view.end(), dstView.begin());
                syncWithCuda();
            }
            copyTotal += stopwatch.printAndReset("copy", '\t');
            for(std::byte* blob : dstView.storageBlobs)
                thrust::device_free(thrust::device_ptr<std::byte>{blob});
        }

        {
            auto dstView = llama::allocView(mapping, thrustDeviceAlloc);
            Stopwatch stopwatch;
            if constexpr(usePSTL)
                std::copy_if(exec, view.begin(), view.end(), dstView.begin(), Predicate{});
            else
            {
                thrust::copy_if(thrust::device, view.begin(), view.end(), dstView.begin(), Predicate{});
                syncWithCuda();
            }
            copyIfTotal += stopwatch.printAndReset("copy_if", '\t');
            for(std::byte* blob : dstView.storageBlobs)
                thrust::device_free(thrust::device_ptr<std::byte>{blob});
        }

        {
            Stopwatch stopwatch;
            if constexpr(usePSTL)
                std::remove_if(exec, view.begin(), view.end(), Predicate{});
            else
            {
                thrust::remove_if(thrust::device, view.begin(), view.end(), Predicate{});
                syncWithCuda();
            }
            removeIfTotal += stopwatch.printAndReset("remove_if", '\t');
        }

        //{
        //    Stopwatch stopwatch;
        //    if constexpr(usePSTL)
        //        std::sort(std::execution::par, view.begin(), view.end(), Less{});
        //    else
        //    {
        //        thrust::sort(thrust::device, view.begin(), view.end(), Less{});
        //        syncWithCuda();
        //    }
        //    sortTotal += stopwatch.printAndReset("sort", '\t');
        //    if(!thrust::is_sorted(thrust::device, view.begin(), view.end(), Less{}))
        //        std::cerr << "VALIDATION FAILED\n";
        //}

        std::cout << std::endl;
    }
    plotFile << std::quoted(mappingName) << '\t';
    plotFile << tabulateTotal / REPETITIONS << '\t';
    // plotFile << shuffleTotal / REPETITIONS << '\t';
    plotFile << forEachTotal / REPETITIONS << '\t';
    plotFile << transformTotal / REPETITIONS << '\t';
    plotFile << transformScanTotal / REPETITIONS << '\t';
    plotFile << transformReduceTotal / REPETITIONS << '\t';
    plotFile << copyTotal / REPETITIONS << '\t';
    plotFile << copyIfTotal / REPETITIONS << '\t';
    plotFile << removeIfTotal / REPETITIONS << '\n';
    // plotFile << sortTotal / REPETITIONS << '\n';

    if constexpr(!thrustUsesCuda)
        // make sure the compiler does not optimize away the benchmark
        for(std::byte* blob : view.storageBlobs)
            sinkp = reinterpret_cast<void*>(blob);

    for(std::byte* blob : view.storageBlobs)
        thrust::device_free(thrust::device_ptr<std::byte>{blob});
}

auto main(int argc, char* argv[]) -> int
try
{
    const auto plotFileName = argc == 1 ? "thrust.sh" : argv[1];

    const auto deviceName = []
    {
        if constexpr(usePSTL)
        {
            if constexpr(std::is_same_v<decltype(exec), decltype(std::execution::seq)>)
                return "STL";
            else
                return "PSTL";
        }
        else
            return "thrust/" BOOST_PP_STRINGIZE(__THRUST_DEVICE_SYSTEM_NAMESPACE);
    }();

    fmt::print(
        "{}Mi particles ({}MiB) with {}\n",
        N / 1024 / 1024,
        N * llama::sizeOf<RecordDim> / 1024 / 1024,
        deviceName);

    // static_assert(llama::mapping::MinAlignedOne<llama::ArrayDims<0>, Particle>{}.blobSize(0) == 16);
    //[[maybe_unused]] const auto v = llama::allocViewStack<0, Particle>();
    // static_assert(sizeof(v.storageBlobs) == 16);
    // static_assert(sizeof(v) == 16);
    //[[maybe_unused]] const auto p = llama::One<Particle>{};
    // static_assert(sizeof(p) == 16);

    // static_assert(sizeof(Struct) == 4);
    // std::cout << "alignof One " << alignof(llama::One<RecordDim>) << "\n";
    // static_assert(alignof(llama::One<Particle>) == 4);

    //{
    //    Stopwatch stopwatch;
    //    thrust::device_vector<Struct> v(N);
    //    thrust::sort(thrust::device, v.begin(), v.end());
    //    stopwatch.printAndReset("Struct sort");
    //}
    //{
    //    Stopwatch stopwatch;
    //    thrust::device_vector<llama::One<RecordDim>> v(N);
    //    thrust::sort(thrust::device, v.begin(), v.end(), Less{});
    //    stopwatch.printAndReset("One sort");
    //}

    const auto numThreads = static_cast<std::size_t>(omp_get_max_threads());
    const char* affinity = std::getenv("GOMP_CPU_AFFINITY"); // NOLINT(concurrency-mt-unsafe)
    affinity = affinity == nullptr ? "NONE - PLEASE PIN YOUR THREADS!" : affinity;

    std::ofstream plotFile{plotFileName};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << fmt::format(
        R"aa(#!/usr/bin/env -S gnuplot -p
# threads: {} affinity: {}
set title "thrust {}Mi particles ({}MiB) with {} on {}"
set style data histograms
set style fill solid
set xtics rotate by 45 right
set key out top center maxrows 3
set yrange [0:*]
set ylabel "runtime [s]"
$data << EOD
""	"tabulate"	"for\\\_each"	"transform"	"transform\\\_scan"	"transform\\\_reduce"	"copy"	"copy\\\_if"	"remove\\\_if"
)aa",
        numThreads,
        affinity,
        N / 1024 / 1024,
        N * llama::sizeOf<RecordDim> / 1024 / 1024,
        deviceName,
        common::hostname(),
        shuffle_frac);

    run<0>(plotFile);
    run<1>(plotFile);
    run<2>(plotFile);
    run<3>(plotFile);
    run<4>(plotFile);
    run<5>(plotFile);
    run<6>(plotFile);

    plotFile << R"(EOD
plot for [COL=2:9] $data using COL:xtic(1) ti col
)";
    fmt::print("Plot with: ./{}\n", plotFileName);
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

// Copyright CERN; jblomer@cern.ch
// Modified by: Bernhard Manfred Gruber

// This example is based on an example provided by the LHCB experiment at CERN. The analysis searches for a difference
// between the behavior of matter and anti-matter. The project analyses B-meson decays to three charged particles.
// A project notebook is provided by LHCB here: http://opendata.cern.ch/record/4902 and on GitHub:
// https://github.com/lhcb/opendata-project.

// A walkthrough of the concrete analysis is here:
// https://github.com/lhcb/opendata-project/blob/master/LHCb_Open_Data_Project.ipynb

#include <ROOT/RNTuple.hxx>
#include <TApplication.h>
#include <TCanvas.h>
#include <TH1D.h>
#include <TRootCanvas.h>
#include <TStyle.h>
#include <TSystem.h>
#include <chrono>
#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <llama/llama.hpp>
#include <omp.h>
#include <string>

// for multithreading, specify OpenMP thread affinity, e.g.:
// OMP_NUM_THREADS=... OMP_PLACES=cores OMP_PROC_BIND=true llama-root-lhcb_analysis

namespace
{
    constexpr auto analysisRepetitions = 100;
    constexpr auto analysisRepetitionsInstrumentation = 1; // costly, so run less often

    // clang-format off
    struct H1isMuon{};
    struct H2isMuon{};
    struct H3isMuon{};

    struct H1PX{};
    struct H1PY{};
    struct H1PZ{};
    struct H1ProbK{};
    struct H1ProbPi{};

    struct H2PX{};
    struct H2PY{};
    struct H2PZ{};
    struct H2ProbK{};
    struct H2ProbPi{};

    struct H3PX{};
    struct H3PY{};
    struct H3PZ{};
    struct H3ProbK{};
    struct H3ProbPi{};
    // clang-format on

    using RecordDim = llama::Record<
        llama::Field<H1isMuon, int>,
        llama::Field<H2isMuon, int>,
        llama::Field<H3isMuon, int>,
        llama::Field<H1PX, double>,
        llama::Field<H1PY, double>,
        llama::Field<H1PZ, double>,
        llama::Field<H1ProbK, double>,
        llama::Field<H1ProbPi, double>,
        llama::Field<H2PX, double>,
        llama::Field<H2PY, double>,
        llama::Field<H2PZ, double>,
        llama::Field<H2ProbK, double>,
        llama::Field<H2ProbPi, double>,
        llama::Field<H3PX, double>,
        llama::Field<H3PY, double>,
        llama::Field<H3PZ, double>,
        llama::Field<H3ProbK, double>,
        llama::Field<H3ProbPi, double>>;

    namespace RE = ROOT::Experimental;

    template<typename Mapping>
    auto convertRNTupleToLLAMA(const std::string& path)
    {
        auto begin = std::chrono::steady_clock::now();

        auto ntuple = RE::RNTupleReader::Open(RE::RNTupleModel::Create(), "DecayTree", path);
        //        try
        //        {
        //            ntuple->PrintInfo(ROOT::Experimental::ENTupleInfo::kStorageDetails);
        //        }
        //        catch(const std::exception& e)
        //        {
        //            fmt::print("PrintInfo error: {}", e.what());
        //        }

        auto view = llama::allocViewUninitialized(Mapping{typename Mapping::ArrayExtents{ntuple->GetNEntries()}});

        auto viewH1IsMuon = ntuple->GetView<int>("H1_isMuon");
        auto viewH2IsMuon = ntuple->GetView<int>("H2_isMuon");
        auto viewH3IsMuon = ntuple->GetView<int>("H3_isMuon");

        auto viewH1PX = ntuple->GetView<double>("H1_PX");
        auto viewH1PY = ntuple->GetView<double>("H1_PY");
        auto viewH1PZ = ntuple->GetView<double>("H1_PZ");
        auto viewH1ProbK = ntuple->GetView<double>("H1_ProbK");
        auto viewH1ProbPi = ntuple->GetView<double>("H1_ProbPi");

        auto viewH2PX = ntuple->GetView<double>("H2_PX");
        auto viewH2PY = ntuple->GetView<double>("H2_PY");
        auto viewH2PZ = ntuple->GetView<double>("H2_PZ");
        auto viewH2ProbK = ntuple->GetView<double>("H2_ProbK");
        auto viewH2ProbPi = ntuple->GetView<double>("H2_ProbPi");

        auto viewH3PX = ntuple->GetView<double>("H3_PX");
        auto viewH3PY = ntuple->GetView<double>("H3_PY");
        auto viewH3PZ = ntuple->GetView<double>("H3_PZ");
        auto viewH3ProbK = ntuple->GetView<double>("H3_ProbK");
        auto viewH3ProbPi = ntuple->GetView<double>("H3_ProbPi");

        for(auto i : ntuple->GetEntryRange())
        {
            auto&& event = view(i);
            event(H1isMuon{}) = viewH1IsMuon(i);
            event(H2isMuon{}) = viewH2IsMuon(i);
            event(H3isMuon{}) = viewH3IsMuon(i);

            // a few sanity checks in case we mess up with the bitpacking
            assert(event(H1isMuon{}) != viewH1IsMuon(i));
            assert(event(H2isMuon{}) != viewH2IsMuon(i));
            assert(event(H3isMuon{}) != viewH3IsMuon(i));

            event(H1PX{}) = viewH1PX(i);
            event(H1PY{}) = viewH1PY(i);
            event(H1PZ{}) = viewH1PZ(i);
            event(H1ProbK{}) = viewH1ProbK(i);
            event(H1ProbPi{}) = viewH1ProbPi(i);

            event(H2PX{}) = viewH2PX(i);
            event(H2PY{}) = viewH2PY(i);
            event(H2PZ{}) = viewH2PZ(i);
            event(H2ProbK{}) = viewH2ProbK(i);
            event(H2ProbPi{}) = viewH2ProbPi(i);

            event(H3PX{}) = viewH3PX(i);
            event(H3PY{}) = viewH3PY(i);
            event(H3PZ{}) = viewH3PZ(i);
            event(H3ProbK{}) = viewH3ProbK(i);
            event(H3ProbPi{}) = viewH3ProbPi(i);
        }

        const auto duration
            = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count();

        return std::tuple{view, duration};
    }

    auto getP2(double px, double py, double pz) -> double
    {
        return px * px + py * py + pz * pz;
    }

    constexpr double kaonMassMeV = 493.677;

    auto getKE(double px, double py, double pz) -> double
    {
        const double p2 = getP2(px, py, pz);
        return std::sqrt(p2 + kaonMassMeV * kaonMassMeV);
    }

    constexpr double probKCut = 0.5;
    constexpr double probPiCut = 0.5;

    template<typename View>
    auto analysis(View& view, const std::string& mappingName)
    {
        auto hists = std::vector<TH1D>(omp_get_max_threads(), TH1D("B_mass", mappingName.c_str(), 500, 5050, 5500));

        auto begin = std::chrono::steady_clock::now();
        const RE::NTupleSize_t n = view.extents()[0];
#pragma omp parallel for
        for(RE::NTupleSize_t i = 0; i < n; i++)
        {
            auto&& event = view[i];
            if(event(H1isMuon{}))
                continue;
            if(event(H2isMuon{}))
                continue;
            if(event(H3isMuon{}))
                continue;

            if(event(H1ProbK{}) < probKCut)
                continue;
            if(event(H2ProbK{}) < probKCut)
                continue;
            if(event(H3ProbK{}) < probKCut)
                continue;

            if(event(H1ProbPi{}) > probPiCut)
                continue;
            if(event(H2ProbPi{}) > probPiCut)
                continue;
            if(event(H3ProbPi{}) > probPiCut)
                continue;

            const double h1px = event(H1PX{});
            const double h1py = event(H1PY{});
            const double h1pz = event(H1PZ{});
            const double h2px = event(H2PX{});
            const double h2py = event(H2PY{});
            const double h2pz = event(H2PZ{});
            const double h3px = event(H3PX{});
            const double h3py = event(H3PY{});
            const double h3pz = event(H3PZ{});

            const double bpx = h1px + h2px + h3px;
            const double bpy = h1py + h2py + h3py;
            const double bpz = h1pz + h2pz + h3pz;
            const double bp2 = getP2(bpx, bpy, bpz);
            const double k1e = getKE(h1px, h1py, h1pz);
            const double k2e = getKE(h2px, h2py, h2pz);
            const double k3e = getKE(h3px, h3py, h3pz);
            const double be = k1e + k2e + k3e;
            const double bmass = std::sqrt(be * be - bp2);

            hists[omp_get_thread_num()].Fill(bmass);
        }
        const auto duration
            = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin);

        for(std::size_t i = 1; i < hists.size(); i++)
            hists[0].Add(&hists[i]);

        return std::tuple{hists[0], duration};
    }

    const auto histogramFolder = std::string("lhcb/histograms");
    const auto layoutsFolder = std::string("lhcb/layouts");
    const auto heatmapFolder = std::string("lhcb/heatmaps");

    void save(TH1D& h, const std::string& mappingName)
    {
        const auto file = std::filesystem::path(histogramFolder + "/" + mappingName + ".png");
        std::filesystem::create_directories(file.parent_path());
        auto c = TCanvas("c", "", 800, 700);
        h.GetXaxis()->SetTitle("m_{KKK} [MeV/c^{2}]");
        h.DrawCopy();
        c.Print(file.c_str());
        // c.Modified();
        // c.Update();
        // auto app = TApplication("", nullptr, nullptr);
        // static_cast<TRootCanvas*>(c.GetCanvasImp())
        //     ->Connect("CloseWindow()", "TApplication", gApplication, "Terminate()");
        // app.Run(true);
    }

    // reference results from single threaded run
    constexpr auto expectedEntries = 23895;
    constexpr auto expectedMean = 5262.231219944131;
    constexpr auto expectedStdDev = 75.02283561602752;

    using AoS = llama::mapping::AoS<llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>, RecordDim>;
    using AoSoA8 = llama::mapping::AoSoA<llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>, RecordDim, 8>;
    using AoSoA16 = llama::mapping::AoSoA<llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>, RecordDim, 16>;
    using SoAASB = llama::mapping::AlignedSingleBlobSoA<llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>, RecordDim>;
    using SoAMB = llama::mapping::MultiBlobSoA<llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>, RecordDim>;

    using AoSHeatmap = llama::mapping::Heatmap<AoS>;
    using AoSFieldAccessCount = llama::mapping::FieldAccessCount<AoS>;

    using boost::mp11::mp_list;

    using AoS_Floats = llama::mapping::ChangeType<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        RecordDim,
        llama::mapping::BindAoS<>::fn,
        mp_list<mp_list<double, float>>>;

    using SoAMB_Floats = llama::mapping::ChangeType<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        RecordDim,
        llama::mapping::BindSoA<llama::mapping::Blobs::OnePerField>::fn,
        mp_list<mp_list<double, float>>>;


    // The CustomN mappings are built upon the observation that HxisMuon is accessed densely, H1PropK 10x less often,
    // H2PropK another 10x less often, and everything else super sparse (around every 300th element).

    using Custom1 = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        RecordDim,
        mp_list<mp_list<H1isMuon>, mp_list<H2isMuon>, mp_list<H3isMuon>, mp_list<H1ProbK>>,
        llama::mapping::AlignedAoS,
        llama::mapping::AlignedAoS,
        true>;

    using Custom2 = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        RecordDim,
        mp_list<mp_list<H1isMuon>, mp_list<H2isMuon>, mp_list<H3isMuon>>,
        llama::mapping::AlignedAoS,
        llama::mapping::
            BindSplit<mp_list<mp_list<H1ProbK>>, llama::mapping::AlignedAoS, llama::mapping::AlignedAoS, true>::fn,
        true>;

    using Custom3 = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        RecordDim,
        mp_list<mp_list<H1isMuon>, mp_list<H2isMuon>, mp_list<H3isMuon>>,
        llama::mapping::AlignedAoS,
        llama::mapping::BindSplit<
            mp_list<mp_list<H1ProbK>>,
            llama::mapping::AlignedAoS,
            llama::mapping::
                BindSplit<mp_list<mp_list<H2ProbK>>, llama::mapping::AlignedAoS, llama::mapping::AlignedAoS, true>::fn,
            true>::fn,
        true>;

    using Custom4 = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        RecordDim,
        mp_list<mp_list<H1isMuon>, mp_list<H2isMuon>, mp_list<H3isMuon>>,
        llama::mapping::AlignedAoS,
        llama::mapping::BindSplit<
            mp_list<mp_list<H1ProbK>, mp_list<H2ProbK>>,
            llama::mapping::AlignedAoS,
            llama::mapping::AlignedAoS,
            true>::fn,
        true>;

    using Custom4Heatmap = llama::mapping::Heatmap<Custom4>;

    using Custom5 = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        RecordDim,
        mp_list<mp_list<H1isMuon>, mp_list<H2isMuon>, mp_list<H3isMuon>>,
        llama::mapping::BindBitPackedIntAoS<llama::Constant<1>, llama::mapping::SignBit::Discard>::fn,
        llama::mapping::BindSplit<
            mp_list<mp_list<H1ProbK>, mp_list<H2ProbK>>,
            llama::mapping::AlignedAoS,
            llama::mapping::AlignedAoS,
            true>::fn,
        true>;

    using Custom6 = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        RecordDim,
        mp_list<mp_list<H1isMuon>, mp_list<H2isMuon>, mp_list<H3isMuon>>,
        llama::mapping::BindBitPackedIntAoS<llama::Constant<1>, llama::mapping::SignBit::Discard>::fn,
        llama::mapping::BindSplit<
            mp_list<mp_list<H1ProbK>, mp_list<H2ProbK>>,
            llama::mapping::BindBitPackedFloatAoS<llama::Constant<6>, llama::Constant<16>>::template fn,
            llama::mapping::BindBitPackedFloatAoS<llama::Constant<6>, llama::Constant<16>>::template fn,
            true>::fn,
        true>;

    using Custom7 = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        RecordDim,
        mp_list<mp_list<H1isMuon>, mp_list<H2isMuon>, mp_list<H3isMuon>>,
        llama::mapping::BindBitPackedIntAoS<llama::Constant<1>, llama::mapping::SignBit::Discard>::fn,
        llama::mapping::BindSplit<
            mp_list<mp_list<H1ProbK>, mp_list<H2ProbK>>,
            llama::mapping::AlignedAoS,
            llama::mapping::BindBitPackedFloatAoS<llama::Constant<6>, llama::Constant<16>>::template fn,
            true>::fn,
        true>;

    using Custom8 = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        RecordDim,
        mp_list<mp_list<H1isMuon>, mp_list<H2isMuon>, mp_list<H3isMuon>>,
        llama::mapping::BindBitPackedIntAoS<llama::Constant<1>, llama::mapping::SignBit::Discard>::fn,
        llama::mapping::BindSplit<
            mp_list<mp_list<H1ProbK>, mp_list<H2ProbK>>,
            llama::mapping::BindChangeType<llama::mapping::BindAoS<>::fn, mp_list<mp_list<double, float>>>::fn,
            llama::mapping::BindBitPackedFloatAoS<llama::Constant<6>, llama::Constant<16>>::template fn,
            true>::fn,
        true>;

    using Custom9 = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        RecordDim,
        mp_list<mp_list<H1isMuon>, mp_list<H2isMuon>, mp_list<H3isMuon>>,
        llama::mapping::BindBitPackedIntAoS<llama::Constant<1>, llama::mapping::SignBit::Discard>::fn,
        llama::mapping::BindSplit<
            mp_list<mp_list<H1ProbK>, mp_list<H2ProbK>>,
            llama::mapping::BindChangeType<llama::mapping::BindAoS<>::fn, mp_list<mp_list<double, float>>>::fn,
            llama::mapping::BindChangeType<llama::mapping::BindAoS<>::fn, mp_list<mp_list<double, float>>>::fn,
            true>::fn,
        true>;

    template<int Exp, int Man>
    using MakeBitpacked = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        RecordDim,
        mp_list<mp_list<H1isMuon>, mp_list<H2isMuon>, mp_list<H3isMuon>>,
        llama::mapping::BindBitPackedIntSoA<llama::Constant<1>, llama::mapping::SignBit::Discard>::fn,
        llama::mapping::BindBitPackedFloatSoA<llama::Constant<Exp>, llama::Constant<Man>>::template fn,
        true>;

    template<typename Mapping>
    auto totalBlobSizes(const Mapping& m) -> std::size_t
    {
        std::size_t total = 0;
        for(std::size_t i = 0; i < Mapping::blobCount; i++)
            total += m.blobSize(i);
        return total;
    }

    template<typename Mapping>
    void saveLayout(const std::filesystem::path& layoutFile)
    {
        std::filesystem::create_directories(layoutFile.parent_path());
        std::ofstream{layoutFile} << llama::toSvg(Mapping{typename Mapping::ArrayExtents{10}});
    }

    template<typename View>
    void saveHeatmap(const View& v, const std::filesystem::path& heatmapFile)
    {
        std::filesystem::create_directories(heatmapFile.parent_path());
        const auto& m = v.mapping();
        m.writeGnuplotDataFileBinary(v.blobs(), std::ofstream{heatmapFile});
        std::ofstream{heatmapFile.parent_path() / "plot.sh"} << View::Mapping::gnuplotScriptBinary;
    }

    template<typename View>
    void clearHeatmap(View& v)
    {
        const auto bc = View::Mapping::blobCount;
        for(int i = bc / 2; i < bc; i++)
            std::memset(&v.blobs()[i][0], 0, v.mapping().blobSize(i));
    }

    template<typename View>
    void clearFieldAccessCounts(View& v)
    {
        v.mapping().fieldHits(v.blobs()) = {};
    }

    template<typename View>
    auto sortView(View& v)
    {
        auto begin = std::chrono::steady_clock::now();
        auto filterResults = [](const auto& e)
        {
            return std::tuple{
                e(H1isMuon{}),
                e(H2isMuon{}),
                e(H3isMuon{}),
                e(H1ProbK{}) < probKCut,
                e(H2ProbK{}) < probKCut,
                e(H3ProbK{}) < probKCut,
                e(H1ProbPi{}) > probPiCut,
                e(H2ProbPi{}) > probPiCut,
                e(H3ProbPi{}) > probPiCut};
        };
        std::sort(
            v.begin(),
            v.end(),
            [&](const auto& a, const auto& b) { return filterResults(a) < filterResults(b); });
        const auto duration
            = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin);
        return duration;
    }

    template<typename View>
    auto countAccessedHeatmapBlocks(const View& v) -> std::size_t
    {
        std::size_t total = 0;
        const auto& m = v.mapping();
        for(std::size_t i = 0; i < View::Mapping::blobCount / 2; i++)
        {
            auto* bh = m.blockHitsPtr(i, v.blobs());
            auto size = m.blockHitsSize(i);
            total += std::count_if(bh, bh + size, [](auto c) { return c > 0; });
        }
        return total;
    }

    template<typename Mapping, bool Sort = false>
    void testAnalysis(const std::string& inputFile, const std::string& mappingName)
    {
        saveLayout<Mapping>(layoutsFolder + "/" + mappingName + ".svg");

        auto [view, conversionTime] = convertRNTupleToLLAMA<Mapping>(inputFile);
        if constexpr(llama::mapping::isFieldAccessCount<Mapping>)
        {
            view.mapping().printFieldHits(view.blobs());
            clearFieldAccessCounts(view);
        }
        if constexpr(llama::mapping::isHeatmap<Mapping>)
        {
            saveHeatmap(view, heatmapFolder + "/" + mappingName + "_conversion.bin");
            clearHeatmap(view);
        }

        std::chrono::microseconds sortTime{};
        if constexpr(Sort)
            sortTime = sortView(view);

        TH1D hist{};
        const auto repetitions = llama::mapping::isFieldAccessCount<Mapping> || llama::mapping::isHeatmap<Mapping>
            ? analysisRepetitionsInstrumentation
            : analysisRepetitions;
        std::chrono::microseconds totalAnalysisTime{};
        for(int i = 0; i < repetitions; i++)
        {
            auto [h, analysisTime] = analysis(view, mappingName);
            if(i == 0)
                hist = h;
            totalAnalysisTime += analysisTime;
        }
        if constexpr(llama::mapping::isFieldAccessCount<Mapping>)
            view.mapping().printFieldHits(view.blobs());
        if constexpr(llama::mapping::isHeatmap<Mapping>)
            saveHeatmap(view, heatmapFolder + "/" + mappingName + "_analysis.bin");
        save(hist, mappingName);
        std::size_t cachlinesLoaded = 0;
        if constexpr(
            !llama::mapping::isHeatmap<Mapping> && !llama::mapping::isFieldAccessCount<Mapping>
            && !llama::hasAnyComputedField<Mapping>)
        {
            // measure cachelines
            auto view2 = llama::allocView(llama::mapping::Heatmap<Mapping, 64>{view.mapping()});
            llama::copy(view, view2);
            clearHeatmap(view2);
            analysis(view2, mappingName);
            cachlinesLoaded = countAccessedHeatmapBlocks(view2);
        }

        const auto mean = hist.GetMean();
        const auto absError = std::abs(mean - expectedMean);
        fmt::print(
            "{:16} {:>15.3f} {:>10.3f} {:>12.3f} {:>4} {:>10.1f} {:>7} {:>6.1f} {:>6.1f} {:>6.1f} {:>6.3f} {:>8}\n",
            "\"" + mappingName + "\"",
            conversionTime / 1000.0,
            sortTime.count() / 1000.0,
            totalAnalysisTime.count() / repetitions / 1000.0,
            repetitions,
            totalBlobSizes(view.mapping()) / 1024.0 / 1024.0,
            hist.GetEntries(),
            mean,
            hist.GetStdDev(),
            std::abs(mean - expectedMean),
            absError / expectedMean,
            cachlinesLoaded);
    }
} // namespace

auto main(int argc, const char* argv[]) -> int
{
    if(argc != 2)
    {
        fmt::print("Please specify location of the LHCB B2HHH RNTuple input file!");
        return 1;
    }

    const auto& inputFile = argv[1];

    gErrorIgnoreLevel = kWarning + 1; // TODO(bgruber): supress warnings that the RNTuple still uses a pre-released
                                      // format. Remove this once RNTuple hits production.

    fmt::print(
        "{:16} {:>15} {:>10} {:>12} {:>4} {:>10} {:>7} {:>6} {:>6} {:>6} {:>6} {:>8}\n",
        "Mapping",
        "RNT->LLAMA(ms)",
        "Sort(ms)",
        "Analysis(ms)",
        "Rep",
        "Size(MiB)",
        "Entries",
        "Mean",
        "StdDev",
        "ErrAbs",
        "ErrRel",
        "$L-load");

    testAnalysis<AoS>(inputFile, "AoS");
    // testAnalysis<AoS, true>(inputFile, "AoS");
    testAnalysis<AoSFieldAccessCount>(inputFile, "AoS FAC"); // also shows how many bytes were needed,
                                                             // which is actually the same for all analyses
    testAnalysis<AoSHeatmap>(inputFile, "AoS Heatmap");
    testAnalysis<AoSoA8>(inputFile, "AoSoA8");
    testAnalysis<AoSoA16>(inputFile, "AoSoA16");
    testAnalysis<SoAASB>(inputFile, "SoA SB A");
    testAnalysis<SoAMB>(inputFile, "SoA MB");
    // testAnalysis<SoAMB, true>(inputFile, "SoA MB S");

    testAnalysis<AoS_Floats>(inputFile, "AoS float");
    testAnalysis<SoAMB_Floats>(inputFile, "SoA MB float");
    // testAnalysis<SoAMB_Floats, true>(inputFile, "SoA MB S float");

    testAnalysis<Custom1>(inputFile, "Custom1");
    testAnalysis<Custom2>(inputFile, "Custom2");
    testAnalysis<Custom3>(inputFile, "Custom3");
    testAnalysis<Custom4>(inputFile, "Custom4");
    testAnalysis<Custom4Heatmap>(inputFile, "Custom4 Heatmap");
    testAnalysis<Custom5>(inputFile, "Custom5");
    testAnalysis<Custom5, true>(inputFile, "Custom5 S");
    testAnalysis<Custom6>(inputFile, "Custom6");
    testAnalysis<Custom6, true>(inputFile, "Custom6 S");
    testAnalysis<Custom7>(inputFile, "Custom7");
    testAnalysis<Custom7, true>(inputFile, "Custom7 S");
    testAnalysis<Custom8>(inputFile, "Custom8");
    testAnalysis<Custom8, true>(inputFile, "Custom8 S");
    testAnalysis<Custom9>(inputFile, "Custom9");
    testAnalysis<Custom9, true>(inputFile, "Custom9 S");

    constexpr auto fullExp = 11;
    constexpr auto fullMan = 52;
    testAnalysis<MakeBitpacked<fullExp, fullMan>>(inputFile, fmt::format("BP SoA {}e{}", fullMan, fullExp));

    // using namespace boost::mp11;
    // mp_for_each<mp_reverse<mp_drop_c<mp_iota_c<fullExp>, 1>>>(
    //     [&](auto ic)
    //     {
    //         constexpr auto exp = decltype(ic)::value;
    //         testAnalysis<MakeBitpacked<exp, fullMan>>(inputFile, fmt::format("BP SoA {}e{}", fullMan, exp));
    //     });
    // mp_for_each<mp_reverse<mp_drop_c<mp_iota_c<fullMan>, 1>>>(
    //     [&](auto ic)
    //     {
    //         constexpr auto man = decltype(ic)::value;
    //         testAnalysis<MakeBitpacked<fullExp, man>>(inputFile, fmt::format("BP SoA {}e{}", man, fullExp));
    //     });

    // we typically observe wrong results at exp < 6, and man < 16
    testAnalysis<MakeBitpacked<6, 16>>(inputFile, "BP SoA 16e6");

    return 0;
}

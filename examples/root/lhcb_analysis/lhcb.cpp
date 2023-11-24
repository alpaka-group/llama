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
    constexpr auto analysisRepetitionsInstrumentation
        = 0; // costly, so turned off by default, use 1 for FieldAccessCounts and Heatmap
    constexpr auto estimateLoadedCachelines = false;

    // clang-format off
    // struct BFlightDistance{};
    // struct BVertexChi2{};

    struct H1{} h1;
    struct H2{} h2;
    struct H3{} h3;

    // struct Charge{} charge;
    // struct IpChi2{} ipChi2;
    struct PX{} px;
    struct PY{} py;
    struct PZ{} pz;
    struct ProbK{} probK;
    struct ProbPi{} probPi;
    struct IsMuon{} isMuon;
    // clang-format on

    // Only needed data is loaded and represented in the LLAMA view. This is also the default behavior of ROOT's
    // RDataFrame and handwritten analyses. Only used columns are loaded.
    using H = llama::Record<
        // llama::Field<Charge, int>,
        // llama::Field<IpChi2, double>,
        llama::Field<PX, double>,
        llama::Field<PY, double>,
        llama::Field<PZ, double>,
        llama::Field<ProbK, double>,
        llama::Field<ProbPi, double>,
        llama::Field<IsMuon, int>>;

    using Event = llama::Record<
        // llama::Field<BFlightDistance, double>,
        // llama::Field<BVertexChi2, double>,
        llama::Field<H1, H>,
        llama::Field<H2, H>,
        llama::Field<H3, H>>;

    namespace RE = ROOT::Experimental;

    template<typename Mapping>
    auto convertRNTupleToLLAMA(std::string_view path, std::string_view treeName)
    {
        auto begin = std::chrono::steady_clock::now();

        auto ntuple = RE::RNTupleReader::Open(RE::RNTupleModel::Create(), treeName, path);
        //        try
        //        {
        //            ntuple->PrintInfo(ROOT::Experimental::ENTupleInfo::kStorageDetails);
        //        }
        //        catch(const std::exception& e)
        //        {
        //            fmt::print("PrintInfo error: {}", e.what());
        //        }

        auto view = llama::allocViewUninitialized(Mapping{typename Mapping::ArrayExtents{ntuple->GetNEntries()}});

        llama::forEachLeafCoord<Event>(
            [&]<typename RecordCoord>(RecordCoord rc)
            {
                using Type = llama::GetType<Event, RecordCoord>;
                auto columnName = std::string(llama::prettyRecordCoord<Event>(rc));
                for(char& c : columnName)
                    if(c == '.')
                        c = '_';
                if(columnName[3] == 'I')
                    columnName[3] = 'i';
                auto columnView = ntuple->GetView<Type>(columnName);
                for(auto i : ntuple->GetEntryRange())
                    view(i)(rc) = columnView(i);
            });
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
            if(event(h1)(isMuon))
                continue;
            if(event(h2)(isMuon))
                continue;
            if(event(h3)(isMuon))
                continue;

            if(event(h1)(probK) < probKCut)
                continue;
            if(event(h2)(probK) < probKCut)
                continue;
            if(event(h3)(probK) < probKCut)
                continue;

            if(event(h1)(probPi) > probPiCut)
                continue;
            if(event(h2)(probPi) > probPiCut)
                continue;
            if(event(h3)(probPi) > probPiCut)
                continue;

            const double h1px = event(h1)(px);
            const double h1py = event(h1)(py);
            const double h1pz = event(h1)(pz);
            const double h2px = event(h2)(px);
            const double h2py = event(h2)(py);
            const double h2pz = event(h2)(pz);
            const double h3px = event(h3)(px);
            const double h3py = event(h3)(py);
            const double h3pz = event(h3)(pz);

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

    const auto histogramFolder = std::filesystem::path("lhcb/histograms");
    const auto layoutsFolder = std::filesystem::path("lhcb/layouts");
    const auto heatmapFolder = std::filesystem::path("lhcb/heatmaps");

    void saveHist(TH1D& h, const std::string& mappingName)
    {
        std::filesystem::create_directories(histogramFolder);
        auto c = TCanvas("c", "", 800, 700);
        h.GetXaxis()->SetTitle("m_{KKK} [MeV/c^{2}]");
        h.DrawCopy("", "");
        c.Print((histogramFolder / (mappingName + ".png")).c_str());
        c.Print((histogramFolder / (mappingName + ".pdf")).c_str());
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

    using AoS = llama::mapping::AoS<llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>, Event>;
    using AoSoA8 = llama::mapping::AoSoA<llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>, Event, 8>;
    using AoSoA16 = llama::mapping::AoSoA<llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>, Event, 16>;
    using SoAASB = llama::mapping::AlignedSingleBlobSoA<llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>, Event>;
    using SoAMB = llama::mapping::MultiBlobSoA<llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>, Event>;

    using AoSHeatmap = llama::mapping::Heatmap<AoS>;
    using AoSFieldAccessCount = llama::mapping::FieldAccessCount<AoS>;

    using boost::mp11::mp_list;

    using AoS_Floats = llama::mapping::ChangeType<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        Event,
        llama::mapping::BindAoS<>::fn,
        mp_list<mp_list<double, float>>>;

    using SoAMB_Floats = llama::mapping::ChangeType<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        Event,
        llama::mapping::BindSoA<llama::mapping::Blobs::OnePerField>::fn,
        mp_list<mp_list<double, float>>>;


    // The CustomN mappings are built upon the observation that HxisMuon is accessed densely, H1PropK 10x less often,
    // H2PropK another 10x less often, and everything else super sparse (around every 300th element).

    using Custom1 = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        Event,
        mp_list<mp_list<H1, IsMuon>, mp_list<H2, IsMuon>, mp_list<H3, IsMuon>, mp_list<H1, ProbK>>,
        llama::mapping::AlignedAoS,
        llama::mapping::AlignedAoS,
        true>;

    using Custom2 = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        Event,
        mp_list<mp_list<H1, IsMuon>, mp_list<H2, IsMuon>, mp_list<H3, IsMuon>>,
        llama::mapping::AlignedAoS,
        llama::mapping::
            BindSplit<mp_list<mp_list<H1, ProbK>>, llama::mapping::AlignedAoS, llama::mapping::AlignedAoS, true>::fn,
        true>;

    using Custom3 = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        Event,
        mp_list<mp_list<H1, IsMuon>, mp_list<H2, IsMuon>, mp_list<H3, IsMuon>>,
        llama::mapping::AlignedAoS,
        llama::mapping::BindSplit<
            mp_list<mp_list<H1, ProbK>>,
            llama::mapping::AlignedAoS,
            llama::mapping::
                BindSplit<mp_list<mp_list<H2, ProbK>>, llama::mapping::AlignedAoS, llama::mapping::AlignedAoS, true>::
                    fn,
            true>::fn,
        true>;

    using Custom4 = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        Event,
        mp_list<mp_list<H1, IsMuon>, mp_list<H2, IsMuon>, mp_list<H3, IsMuon>>,
        llama::mapping::AlignedAoS,
        llama::mapping::BindSplit<
            mp_list<mp_list<H1, ProbK>, mp_list<H2, ProbK>>,
            llama::mapping::AlignedAoS,
            llama::mapping::AlignedAoS,
            true>::fn,
        true>;

    using Custom1_3_H1ProbK_float = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        Event,
        mp_list<mp_list<H1, IsMuon>, mp_list<H2, IsMuon>, mp_list<H3, IsMuon>, mp_list<H1, ProbK>>,
        llama::mapping::BindChangeType<llama::mapping::BindAoS<>::fn, mp_list<mp_list<double, float>>>::fn,
        llama::mapping::
            BindSplit<mp_list<mp_list<H2, ProbK>>, llama::mapping::AlignedAoS, llama::mapping::AlignedAoS, true>::fn,
        true>;

    using Custom4Heatmap = llama::mapping::Heatmap<Custom4>;

    using Custom5 = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        Event,
        mp_list<mp_list<H1, IsMuon>, mp_list<H2, IsMuon>, mp_list<H3, IsMuon>>,
        llama::mapping::BindBitPackedIntAoS<llama::Constant<1>, llama::mapping::SignBit::Discard>::fn,
        llama::mapping::BindSplit<
            mp_list<mp_list<H1, ProbK>, mp_list<H2, ProbK>>,
            llama::mapping::AlignedAoS,
            llama::mapping::AlignedAoS,
            true>::fn,
        true>;

    template<std::size_t ManBits = 16>
    using Custom6 = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        Event,
        mp_list<mp_list<H1, IsMuon>, mp_list<H2, IsMuon>, mp_list<H3, IsMuon>>,
        llama::mapping::BindBitPackedIntAoS<llama::Constant<1>, llama::mapping::SignBit::Discard>::fn,
        llama::mapping::BindSplit<
            mp_list<mp_list<H1, ProbK>, mp_list<H2, ProbK>>,
            llama::mapping::BindBitPackedFloatAoS<llama::Constant<6>, llama::Constant<ManBits>>::template fn,
            llama::mapping::BindBitPackedFloatAoS<llama::Constant<6>, llama::Constant<ManBits>>::template fn,
            true>::template fn,
        true>;

    using Custom7 = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        Event,
        mp_list<mp_list<H1, IsMuon>, mp_list<H2, IsMuon>, mp_list<H3, IsMuon>>,
        llama::mapping::BindBitPackedIntAoS<llama::Constant<1>, llama::mapping::SignBit::Discard>::fn,
        llama::mapping::BindSplit<
            mp_list<mp_list<H1, ProbK>, mp_list<H2, ProbK>>,
            llama::mapping::AlignedAoS,
            llama::mapping::BindBitPackedFloatAoS<llama::Constant<6>, llama::Constant<16>>::template fn,
            true>::fn,
        true>;

    template<std::size_t ManBits = 16>
    using Custom8 = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        Event,
        mp_list<mp_list<H1, IsMuon>, mp_list<H2, IsMuon>, mp_list<H3, IsMuon>>,
        llama::mapping::BindBitPackedIntAoS<llama::Constant<1>, llama::mapping::SignBit::Discard>::fn,
        llama::mapping::BindSplit<
            mp_list<mp_list<H1, ProbK>, mp_list<H2, ProbK>>,
            llama::mapping::BindChangeType<llama::mapping::BindAoS<>::fn, mp_list<mp_list<double, float>>>::fn,
            llama::mapping::BindBitPackedFloatAoS<llama::Constant<6>, llama::Constant<ManBits>>::template fn,
            true>::template fn,
        true>;

    using Custom9 = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        Event,
        mp_list<mp_list<H1, IsMuon>, mp_list<H2, IsMuon>, mp_list<H3, IsMuon>>,
        llama::mapping::BindBitPackedIntAoS<llama::Constant<1>, llama::mapping::SignBit::Discard>::fn,
        llama::mapping::BindSplit<
            mp_list<mp_list<H1, ProbK>, mp_list<H2, ProbK>>,
            llama::mapping::BindChangeType<llama::mapping::BindAoS<>::fn, mp_list<mp_list<double, float>>>::fn,
            llama::mapping::BindChangeType<llama::mapping::BindAoS<>::fn, mp_list<mp_list<double, float>>>::fn,
            true>::fn,
        true>;

    template<int Exp, int Man>
    using MakeBitpacked = llama::mapping::Split<
        llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>,
        Event,
        mp_list<mp_list<H1, IsMuon>, mp_list<H2, IsMuon>, mp_list<H3, IsMuon>>,
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
        std::filesystem::create_directories(layoutsFolder);
        std::ofstream{layoutsFolder / layoutFile} << llama::toSvg(Mapping{typename Mapping::ArrayExtents{3}}, 32);
    }

    template<typename View>
    void saveHeatmap(const View& v, const std::filesystem::path& heatmapFile)
    {
        std::filesystem::create_directories(heatmapFolder);
        const auto& m = v.mapping();
        m.writeGnuplotDataFileBinary(v.blobs(), std::ofstream{heatmapFolder / heatmapFile});
        std::ofstream{heatmapFolder / "plot.sh"} << View::Mapping::gnuplotScriptBinary;
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
                e(h1)(isMuon),
                e(h2)(isMuon),
                e(h3)(isMuon),
                e(h1)(probK) < probKCut,
                e(h2)(probK) < probKCut,
                e(h3)(probK) < probKCut,
                e(h1)(probPi) > probPiCut,
                e(h2)(probPi) > probPiCut,
                e(h3)(probPi) > probPiCut};
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
    void testAnalysis(std::string_view inputFile, std::string_view treeName, const std::string& mappingName)
    {
        const auto repetitions = llama::mapping::isFieldAccessCount<Mapping> || llama::mapping::isHeatmap<Mapping>
            ? analysisRepetitionsInstrumentation
            : analysisRepetitions;
        if(repetitions == 0)
            return;
        saveLayout<Mapping>(mappingName + ".svg");

        auto [view, conversionTime] = convertRNTupleToLLAMA<Mapping>(inputFile, treeName);
        if constexpr(llama::mapping::isFieldAccessCount<Mapping>)
        {
            view.mapping().printFieldHits(view.blobs());
            clearFieldAccessCounts(view);
        }
        if constexpr(llama::mapping::isHeatmap<Mapping>)
        {
            saveHeatmap(view, heatmapFolder / (mappingName + "_conversion.bin"));
            clearHeatmap(view);
        }

        std::chrono::microseconds sortTime{}; // NOLINT(misc-const-correctness)
        if constexpr(Sort)
            sortTime = sortView(view);

        TH1D hist{};
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
            saveHeatmap(view, mappingName + "_analysis.bin");
        saveHist(hist, mappingName);
        std::size_t cachlinesLoaded = 0;
        if constexpr(
            estimateLoadedCachelines && !llama::mapping::isHeatmap<Mapping>
            && !llama::mapping::isFieldAccessCount<Mapping> && !llama::hasAnyComputedField<Mapping>)
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
            "{:13} {:>9.3f} {:>9.3f} {:>9.3f} {:>4} {:>10.1f} {:>7} {:>6.1f} {:>6.1f} {:>6.1f} {:>6.3f} {:>8}\n",
            mappingName,
            conversionTime / 1000.0,
            static_cast<double>(sortTime.count()) / 1000.0,
            static_cast<double>(totalAnalysisTime.count()) / repetitions / 1000.0,
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
    if(argc != 2 && argc != 3)
    {
        fmt::print(
            "Invalid command line arguments. Usage:\n\n"
            "{} <inputfile> [treename, default: \"DecayTree\"]\n",
            argv[0]);
        return 1;
    }

    const auto& inputFile = argv[1];
    const auto treeName = std::string_view(argc == 3 ? argv[2] : "DecayTree");

    gErrorIgnoreLevel = kWarning + 1; // TODO(bgruber): supress warnings that the RNTuple still uses a pre-released
                                      // format. Remove this once RNTuple hits production.

    fmt::print(
        "{:13} {:>9} {:>9} {:>9} {:>4} {:>10} {:>7} {:>6} {:>6} {:>6} {:>6} {:>8}\n",
        "Mapping",
        "Read(ms)",
        "Sort(ms)",
        "Anly(ms)",
        "Rep",
        "Size(MiB)",
        "Entries",
        "Mean",
        "StdDev",
        "ErrAbs",
        "ErrRel",
        "$L-load");

    testAnalysis<AoS>(inputFile, treeName, "AoS");
    // testAnalysis<AoS, true>(inputFile, treeName,"AoS");
    testAnalysis<AoSFieldAccessCount>(inputFile, treeName, "AoS_FAC"); // also shows how many bytes were needed,
                                                                       // which is actually the same for all analyses
    testAnalysis<AoSHeatmap>(inputFile, treeName, "AoS_HM");
    testAnalysis<AoSoA8>(inputFile, treeName, "AoSoA8");
    testAnalysis<AoSoA16>(inputFile, treeName, "AoSoA16");
    testAnalysis<SoAASB>(inputFile, treeName, "SoA_SB_A");
    testAnalysis<SoAMB>(inputFile, treeName, "SoA_MB");
    // testAnalysis<SoAMB, true>(inputFile, treeName,"SoA MB S");

    testAnalysis<AoS_Floats>(inputFile, treeName, "AoS_F");
    testAnalysis<SoAMB_Floats>(inputFile, treeName, "SoA_MB_F");
    // testAnalysis<SoAMB_Floats, true>(inputFile, treeName,"SoA MB S float");

    testAnalysis<Custom1>(inputFile, treeName, "Custom1");
    testAnalysis<Custom2>(inputFile, treeName, "Custom2");
    testAnalysis<Custom3>(inputFile, treeName, "Custom3");
    testAnalysis<Custom4>(inputFile, treeName, "Custom4");
    testAnalysis<Custom4Heatmap>(inputFile, treeName, "Custom4_HM");
    testAnalysis<Custom5>(inputFile, treeName, "Custom5");
    //    testAnalysis<Custom5, true>(inputFile, treeName, "Custom5_S");
    testAnalysis<Custom6<>>(inputFile, treeName, "Custom6");
    //    testAnalysis<Custom6<>, true>(inputFile, treeName, "Custom6_S");
    testAnalysis<Custom7>(inputFile, treeName, "Custom7");
    //    testAnalysis<Custom7, true>(inputFile, treeName, "Custom7_S");
    testAnalysis<Custom8<>>(inputFile, treeName, "Custom8");
    //    testAnalysis<Custom8<>, true>(inputFile, treeName, "Custom8_S");
    testAnalysis<Custom9>(inputFile, treeName, "Custom9");
    //    testAnalysis<Custom9, true>(inputFile, treeName, "Custom9_S");
    testAnalysis<Custom1_3_H1ProbK_float>(inputFile, treeName, "Custom1_3_F");

    constexpr auto fullExp = 11;
    constexpr auto fullMan = 52;
    testAnalysis<MakeBitpacked<fullExp, fullMan>>(inputFile, treeName, fmt::format("BP_SoA_{}e{}", fullMan, fullExp));

    // using namespace boost::mp11;
    // mp_for_each<mp_reverse<mp_iota_c<fullExp>>>(
    //     [&](auto ic)
    //     {
    //         constexpr auto exp = decltype(ic)::value;
    //         testAnalysis<MakeBitpacked<exp, fullMan>>(inputFile, treeName, fmt::format("BP_SoA_{}e{}", fullMan,
    //         exp));
    //     });
    // mp_for_each<mp_reverse<mp_iota_c<fullMan>>>(
    //     [&](auto ic)
    //     {
    //         constexpr auto man = decltype(ic)::value;
    //         testAnalysis<MakeBitpacked<fullExp, man>>(inputFile, treeName, fmt::format("BP_SoA_{}e{}", man,
    //         fullExp));
    //     });

    // we typically observe wrong results at exp < 6, and man < 16
    testAnalysis<MakeBitpacked<6, 16>>(inputFile, treeName, "BP_SoA_16e6");

    // mp_for_each<mp_reverse<mp_iota_c<fullMan + 1>>>(
    //     [&](auto ic)
    //     {
    //         constexpr auto man = decltype(ic)::value;
    //         testAnalysis<Custom8<man>>(inputFile, treeName,fmt::format("Custom8_16e{}", man));
    //     });
    //
    // mp_for_each<mp_reverse<mp_iota_c<fullMan + 1>>>(
    //     [&](auto ic)
    //     {
    //         constexpr auto man = decltype(ic)::value;
    //         testAnalysis<Custom6<man>>(inputFile, treeName,fmt::format("Custom6_16e{}", man));
    //     });

    return 0;
}

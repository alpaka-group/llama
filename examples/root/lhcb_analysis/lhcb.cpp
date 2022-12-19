// Copyright CERN; jblomer@cern.ch
// Modified by: Bernhard Manfred Gruber

#include <ROOT/RNTuple.hxx>
#include <TApplication.h>
#include <TCanvas.h>
#include <TH1D.h>
#include <TRootCanvas.h>
#include <TStyle.h>
#include <TSystem.h>
#include <chrono>
#include <fmt/core.h>
#include <llama/llama.hpp>
#include <string>

namespace
{
    constexpr double kKaonMassMeV = 493.677;

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

        auto ae = llama::ArrayExtentsDynamic<RE::NTupleSize_t, 1>{ntuple->GetNEntries()};
        auto mapping = llama::mapping::AoS{ae, RecordDim{}};
        auto view = llama::allocView(mapping);

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
        fmt::print("RNTuple -> LLAMA view: {}μs\n", duration);

        return view;
    }

    auto GetP2(double px, double py, double pz) -> double
    {
        return px * px + py * py + pz * pz;
    }

    auto GetKE(double px, double py, double pz) -> double
    {
        const double p2 = GetP2(px, py, pz);
        return std::sqrt(p2 + kKaonMassMeV * kKaonMassMeV);
    }

    template<typename View>
    auto analysis(View& view)
    {
        auto hMass = TH1D("B_mass", "", 500, 5050, 5500);

        auto begin = std::chrono::steady_clock::now();
        for(auto i = 0; i < view.mapping().extents()[0]; i++)
        {
            auto&& event = view[i];
            if(event(H1isMuon{}) || event(H2isMuon{}) || event(H3isMuon{}))
                continue;

            constexpr double prob_k_cut = 0.5;
            if(event(H1ProbK{}) < prob_k_cut || event(H2ProbK{}) < prob_k_cut || event(H3ProbK{}) < prob_k_cut)
                continue;

            constexpr double prob_pi_cut = 0.5;
            if(event(H1ProbPi{}) > prob_pi_cut || event(H2ProbPi{}) > prob_pi_cut || event(H3ProbPi{}) > prob_pi_cut)
                continue;

            const double b_px = event(H1PX{}) + event(H2PX{}) + event(H3PX{});
            const double b_py = event(H1PY{}) + event(H2PY{}) + event(H3PY{});
            const double b_pz = event(H1PZ{}) + event(H2PZ{}) + event(H3PZ{});
            const double b_p2 = GetP2(b_px, b_py, b_pz);
            const double k1_E = GetKE(event(H1PX{}), event(H1PY{}), event(H1PZ{}));
            const double k2_E = GetKE(event(H2PX{}), event(H2PY{}), event(H2PZ{}));
            const double k3_E = GetKE(event(H3PX{}), event(H3PY{}), event(H3PZ{}));
            const double b_E = k1_E + k2_E + k3_E;
            const double b_mass = std::sqrt(b_E * b_E - b_p2);
            hMass.Fill(b_mass);
        }
        const auto duration
            = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count();
        fmt::print("Analysis: {}μs\n", duration);

        return hMass;
    }

    void show(TH1D& h)
    {
        auto app = TApplication("", nullptr, nullptr);
        gStyle->SetTextFont(42);
        auto c = TCanvas("c", "", 800, 700);
        h.GetXaxis()->SetTitle("m_{KKK} [MeV/c^{2}]");
        h.DrawCopy();
        c.Modified();
        c.Update();
        static_cast<TRootCanvas*>(c.GetCanvasImp())
            ->Connect("CloseWindow()", "TApplication", gApplication, "Terminate()");
        app.Run();
    }
} // namespace

int main(int argc, const char* argv[])
{
    if(argc != 2)
    {
        fmt::print("Please specify RNTuple input file!");
        return 1;
    }

    const auto& inputFile = argv[1];
    auto view = convertRNTupleToLLAMA(inputFile);
    auto hist = analysis(view);
    show(hist);

    return 0;
}
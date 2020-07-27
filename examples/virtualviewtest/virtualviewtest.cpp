/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

#include "../common/Dummy.hpp"

#include <iostream>
#include <llama/llama.hpp>
#include <math.h>
#include <utility>

#define VIRTUALVIEWTEST_WITH_MINIVIEW 1
#define VIRTUALVIEWTEST_WITH_MINIVIEW_DUMMY 1

/** \file virtualviewtest.cpp
 *  \brief Example for LLAMA showing how to define a virtual view.
 */

namespace virtualviewtest
{
    namespace st
    {
        struct X
        {};
        struct Y
        {};
        struct Z
        {};
        struct Pos
        {};
        struct Vel
        {};
        struct Mom
        {};
    } // namespace st

#ifdef NDEBUG
    using Element = float;
#else
    // Better for check of correctness at the end
    using Element = int;
#endif

    using Particle = llama::DS<
        llama::DE<
            st::Pos,
            llama::DS<
                llama::DE<st::X, Element>,
                llama::DE<st::Y, Element>,
                llama::DE<st::Z, Element>>>,
        llama::DE<st::Mom, Element>,
        llama::DE<
            st::Vel,
            llama::DS<
                llama::DE<st::Z, Element>,
                llama::DE<st::Y, Element>,
                llama::DE<st::X, Element>>>>;

    template<typename T_VirtualDatum>
    struct SqrtFunctor
    {
        template<typename T_OuterCoord, typename T_InnerCoord>
        auto operator()(T_OuterCoord, T_InnerCoord) -> void
        {
            using Coord = typename T_OuterCoord::template Cat<T_InnerCoord>;
            vd(Coord()) *= sqrt(vd(Coord()));
        }
        T_VirtualDatum vd;
    };

    int main(int argc, char ** argv)
    {
        using UD = llama::UserDomain<2>;
        constexpr UD viewSize{4096, 4096};
        constexpr UD miniSize{128, 128};
        using Mapping = llama::mapping::SoA<UD, Particle>;
        using MiniMapping = llama::mapping::SoA<UD, Particle>;
        auto view = llama::Factory<Mapping, llama::allocator::SharedPtr<256>>::
            allocView(Mapping(viewSize));

        for(std::size_t x = 0; x < viewSize[0]; ++x)
            for(std::size_t y = 0; y < viewSize[1]; ++y) view(x, y) = x * y;

        // fast virtual view test
        {
            llama::VirtualView<decltype(view)> virtualView(
                view,
                {23, 42}, // position
                {13, 37} // size
            );
            std::cout << "Created VirtualView at "
                      << "{ " << virtualView.position[0] << ", "
                      << virtualView.position[1] << " } "
                      << "with size "
                      << "{ " << virtualView.size[0] << ", "
                      << virtualView.size[1] << " } " << '\n';
            std::cout << "View"
                      << "( " << virtualView.position[0] << ", "
                      << virtualView.position[1] << " ).pos.x "
                      << "= " << view(virtualView.position)(st::Pos(), st::X())
                      << '\n';
            std::cout << "VirtualView"
                      << "( 0, 0 ).pos.x "
                      << "= " << virtualView({0, 0})(st::Pos(), st::X())
                      << '\n';
            std::cout << "View"
                      << "( " << virtualView.position[0] + 2 << ", "
                      << virtualView.position[1] + 3 << " ).vel.z "
                      << "= "
                      << view(
                             {virtualView.position[0] + 2,
                              virtualView.position[1] + 3})(st::Vel(), st::Z())
                      << '\n';
            std::cout << "VirtualView"
                      << "( 2, 3 ).vel.z "
                      << "= " << virtualView({2, 3})(st::Vel(), st::Z())
                      << '\n';
        }

        constexpr UD iterations{
            (viewSize[0] + miniSize[0] - 1) / miniSize[0],
            (viewSize[1] + miniSize[1] - 1) / miniSize[1]};
        LLAMA_INDEPENDENT_DATA
        for(std::size_t x = 0; x < iterations[0]; ++x) LLAMA_INDEPENDENT_DATA
        for(std::size_t y = 0; y < iterations[1]; ++y)
        {
            const UD validMiniSize{
                (x < iterations[0] - 1) ? miniSize[0]
                                        : (viewSize[0] - 1) % miniSize[0] + 1,
                (y < iterations[1] - 1) ? miniSize[1]
                                        : (viewSize[1] - 1) % miniSize[1] + 1};

            // Create virtual view with size of mini view
            llama::VirtualView<decltype(view)> virtualView(
                view, {x * miniSize[0], y * miniSize[1]}, miniSize);

#if VIRTUALVIEWTEST_WITH_MINIVIEW != 0
            // Create mini view on stack
            auto miniView = llama::Factory<
                MiniMapping,
                llama::allocator::Stack<
                    miniSize[0] * miniSize[1]
                    * llama::SizeOf<Particle>::value>>::
                allocView(MiniMapping(miniSize));

            // Copy data from virtual view to mini view
            LLAMA_INDEPENDENT_DATA
            for(std::size_t a = 0; a < validMiniSize[0]; ++a)
                LLAMA_INDEPENDENT_DATA
            for(std::size_t b = 0; b < validMiniSize[1]; ++b)
                miniView(a, b) = virtualView(a, b);

                // Do something with mini view
#if VIRTUALVIEWTEST_WITH_MINIVIEW_DUMMY != 0
            dummy(static_cast<void *>(&(miniView.blob[0][0])));
#endif // VIRTUALVIEWTEST_WITH_MINIVIEW_DUMMY
#endif // VIRTUALVIEWTEST_WITH_MINIVIEW

            LLAMA_INDEPENDENT_DATA
            for(std::size_t a = 0; a < validMiniSize[0]; ++a)
                LLAMA_INDEPENDENT_DATA
            for(std::size_t b = 0; b < validMiniSize[1]; ++b)
            {
                SqrtFunctor<
#if VIRTUALVIEWTEST_WITH_MINIVIEW != 0
                    decltype(miniView(a, b))>
                    sqrtF{miniView(a, b)};
#else // VIRTUALVIEWTEST_WITH_MINIVIEW
                    decltype(virtualView(a, b))>
                    sqrtF{virtualView(a, b)};
#endif // VIRTUALVIEWTEST_WITH_MINIVIEW
                llama::ForEach<Particle>::apply(sqrtF);
            }

#if VIRTUALVIEWTEST_WITH_MINIVIEW != 0
#if VIRTUALVIEWTEST_WITH_MINIVIEW_DUMMY != 0
            dummy(static_cast<void *>(&(miniView.blob[0][0])));
#endif // VIRTUALVIEWTEST_WITH_MINIVIEW_DUMMY
       // Copy data back
            LLAMA_INDEPENDENT_DATA
            for(std::size_t a = 0; a < validMiniSize[0]; ++a)
                LLAMA_INDEPENDENT_DATA
            for(std::size_t b = 0; b < validMiniSize[1]; ++b)
                virtualView(a, b) = miniView(a, b);
#endif // VIRTUALVIEWTEST_WITH_MINIVIEW
        }
        dummy(static_cast<void *>(&(view.blob[0][0])));

#ifndef NDEBUG
        for(std::size_t x = 0; x < viewSize[0]; ++x)
            for(std::size_t y = 0; y < viewSize[1]; ++y)
            {
                if(view(x, y) != x * y * sqrt(x * y))
                {
                    std::cout << "view( " << x << ", " << y << " ) "
                              << "has unexpected result: \n"
                              << "\tpos.x: " << view(x, y)(st::Pos(), st::X())
                              << "\tpos.y: " << view(x, y)(st::Pos(), st::Y())
                              << "\tpos.z: " << view(x, y)(st::Pos(), st::Z())
                              << "\tvel.x: " << view(x, y)(st::Vel(), st::X())
                              << "\tvel.y: " << view(x, y)(st::Vel(), st::Y())
                              << "\tvel.z: " << view(x, y)(st::Vel(), st::Z())
                              << "\t  mom: " << view(x, y)(st::Mom()) << '\n';
                    // yes.
                    goto end;
                }
            }
    end:
#endif
        return 0;
    }

} // namespace virtualviewtest

int main(int argc, char ** argv)
{
    return virtualviewtest::main(argc, argv);
}

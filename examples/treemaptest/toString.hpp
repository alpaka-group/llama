/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

#pragma once

/** \file treemaptest/toString.hpp
 *  \brief Defining some overloads for the `llama::mapping::tree::ToString`
 *  functor for the treemaptest datum domain uid types.
 */

namespace llama
{
    namespace mapping
    {
        namespace tree
        {
            template<>
            struct ToString<treemaptest::st::Pos>
            {
                auto operator()(const treemaptest::st::Pos) -> std::string
                {
                    return "Pos";
                }
            };

            template<>
            struct ToString<treemaptest::st::X>
            {
                auto operator()(const treemaptest::st::X) -> std::string
                {
                    return "X";
                }
            };

            template<>
            struct ToString<treemaptest::st::Y>
            {
                auto operator()(const treemaptest::st::Y) -> std::string
                {
                    return "Y";
                }
            };

            template<>
            struct ToString<treemaptest::st::Z>
            {
                auto operator()(const treemaptest::st::Z) -> std::string
                {
                    return "Z";
                }
            };

            template<>
            struct ToString<treemaptest::st::Momentum>
            {
                auto operator()(const treemaptest::st::Momentum) -> std::string
                {
                    return "Momentum";
                }
            };

            template<>
            struct ToString<treemaptest::st::Weight>
            {
                auto operator()(const treemaptest::st::Weight) -> std::string
                {
                    return "Weight";
                }
            };

        } // namespace tree

    } // namespace mapping

} // namespace llama

/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

/** \file simpletest.cpp
 *  \brief Simple example for LLAMA showing how to define a user and datum
 *  domain, to create a view and to access the data
 */

#include <iostream>
#include <llama/llama.hpp>
#include <sstream>
#include <utility>
#include <vector>

/// LLAMA uses class names for accessing data in the datum domain. It makes
/// sense to encapsulate these in own namespaces, especially if you are using
/// similar namings in differnt datum domains, e.g. X, Y and Z in different
/// kinds of spaces.

// clang-format off
namespace st
{
    struct Pos{};
    struct X{};
    struct Y{};
    struct Z{};
    struct Momentum{};
    struct Weight{};
    struct Options{};
} // namespace st

/// A datum domain in LLAMA is a type, probably always a
/// \ref llama::DatumStruct, which can be shortend to llama::DS. This takes a
/// template list of all members of this struct-like domain. Every member needs
/// to be a \ref llama::DatumElement, which can itself be shortend to llama::DE.
/// A DatumElement is a list of two elements itself, first the name of the
/// element as type and secondly the element type itself, which may be a
/// nested DatumStruct. A handy shortcut is \ref llama::DatumArray (llama::DA),
/// which defines multuple datum elements with an anonymous name but of the same
/// type.
using Name = llama::DS<
    llama::DE<st::Pos, llama::DS<
        llama::DE<st::X, float>,
        llama::DE<st::Y, float>,
        llama::DE<st::Z, float>>>,
    llama::DE<st::Momentum, llama::DS<
        llama::DE<st::Z, double>,
        llama::DE<st::X, double>>>,
    llama::DE<st::Weight, int>,
    llama::DE<st::Options, bool[4]>>;
// clang-format on

namespace
{
    template <class T>
    auto type(const T& t) -> std::string
    {
        return boost::core::demangle(typeid(t).name());
    }

    /// Prints the coordinates of a given \ref llama::DatumCoord for debugging
    /// and testing purposes
    template <std::size_t... T_coords>
    void printCoords(llama::DatumCoord<T_coords...>)
    {
        (std::cout << ... << T_coords);
    }

    template <typename Out>
    void split(const std::string& s, char delim, Out result)
    {
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, delim))
        {
            *(result++) = item;
        }
    }

    auto split(const std::string& s, char delim) -> std::vector<std::string>
    {
        std::vector<std::string> elems;
        split(s, delim, std::back_inserter(elems));
        return elems;
    }

    auto nSpaces(int n) -> std::string
    {
        std::string result;
        for (int i = 0; i < n; ++i)
            result += " ";
        return result;
    }

    auto addLineBreaks(std::string raw) -> std::string
    {
        using llama::mapping::tree::internal::replace_all;
        replace_all(raw, "<", "<\n");
        replace_all(raw, ", ", ",\n");
        replace_all(raw, " >", ">");
        replace_all(raw, ">", "\n>");
        auto tokens = split(raw, '\n');
        std::string result;
        int indent = 0;
        for (auto t : tokens)
        {
            if (t.back() == '>' || (t.length() > 1 && t[t.length() - 2] == '>'))
                indent -= 4;
            result += nSpaces(indent) + t + "\n";
            if (t.back() == '<')
                indent += 4;
        }
        return result;
    }
} // namespace

/// Example functor for \ref llama::forEachLeaf which can also be used to print the
/// coordinates inside of a datum domain when called.
template <typename VirtualDatum>
struct SetZeroFunctor
{
    template <typename Coord>
    void operator()(Coord coord)
    {
        vd(coord) = 0;
    }
    VirtualDatum vd;
};

auto main() -> int
try
{
    // Defining a two-dimensional user domain
    using UD = llama::ArrayDomain<2>;
    // Setting the run time size of the user domain to 8192 * 8192
    UD udSize{8192, 8192};

    // Printing the domain informations at runtime
    std::cout << "Datum Domain is " << addLineBreaks(type(Name())) << '\n';
    std::cout << "AoS address of (0,100) <0,1>: "
              << llama::mapping::AoS<UD, Name>(udSize).blobNrAndOffset<0, 1>({0, 100}).offset << '\n';
    std::cout << "SoA address of (0,100) <0,1>: "
              << llama::mapping::SoA<UD, Name>(udSize).blobNrAndOffset<0, 1>({0, 100}).offset << '\n';
    std::cout << "sizeOf DatumDomain: " << llama::sizeOf<Name> << '\n';

    std::cout << type(llama::GetCoordFromTags<Name, st::Pos, st::X>()) << '\n';

    // chosing a native struct of array mapping for this simple test example
    using Mapping = llama::mapping::SoA<UD, Name>;

    // Instantiating the mapping with the user domain size
    Mapping mapping(udSize);
    // getting a view with memory from the default allocator
    auto view = allocView(mapping);

    // defining a position in the user domain
    const UD pos{0, 0};

    st::Options Options_;
    const auto Weight_ = st::Weight{};

    // using the position in the user domain and a tree coord or a uid in the
    // datum domain to get the reference to an element in the view
    float& position_x = view(pos)(llama::DatumCoord<0, 0>{});
    double& momentum_z = view[pos](st::Momentum{}, st::Z{});
    int& weight = view[{0, 0}](llama::DatumCoord<2>{});
    int& weight_2 = view(pos)(Weight_);
    bool& options_2 = view[{0, 0}](st::Options())(llama::DatumCoord<2>());
    bool& options_3 = view(pos)(Options_)(llama::DatumCoord<2>());
    // printing the address and distances of the element in the memory. This
    // will change based on the chosen mapping. When array of struct is chosen
    // instead the elements will be much closer than with struct of array.
    std::cout << &position_x << '\n';
    std::cout << &momentum_z << " "
              << reinterpret_cast<std::byte*>(&momentum_z) - reinterpret_cast<std::byte*>(&position_x) << '\n';
    std::cout << &weight << " " << reinterpret_cast<std::byte*>(&weight) - reinterpret_cast<std::byte*>(&momentum_z)
              << '\n';
    std::cout << &options_2 << " " << reinterpret_cast<std::byte*>(&options_2) - reinterpret_cast<std::byte*>(&weight)
              << '\n';

    // iterating over the user domain at run time to do some stuff with the
    // allocated data
    for (size_t x = 0; x < udSize[0]; ++x)
        // telling the compiler that all data in the following loop is
        // independent to each other and thus can be vectorized
        LLAMA_INDEPENDENT_DATA
    for (size_t y = 0; y < udSize[1]; ++y)
    {
        // Defining a functor for a given virtual datum
        SetZeroFunctor<decltype(view(x, y))> szf{view(x, y)};
        // Applying the functor for the sub tree 0,0 (pos.x), so basically
        // only for this element
        llama::forEachLeaf<Name>(szf, llama::DatumCoord<0, 0>{});
        // Applying the functor for the sub tree momentum (0), so basically
        // for momentum.z, and momentum.x
        llama::forEachLeaf<Name>(szf, st::Momentum{});
        // the user domain address can be given as multiple comma separated
        // arguments or as one parameter of type user domain
        view({x, y}) = double(x + y) / double(udSize[0] + udSize[1]);
    }

    for (size_t x = 0; x < udSize[0]; ++x)
        LLAMA_INDEPENDENT_DATA
    for (size_t y = 0; y < udSize[1]; ++y)
    {
        // Showing different options of access data with llama. Internally
        // all do the same data- and mappingwise
        auto datum = view(x, y);
        datum(st::Pos{}, st::X{}) += static_cast<float>(datum(llama::DatumCoord<1, 0>{}));
        datum(st::Pos{}, st::Y{}) += static_cast<float>(datum(llama::DatumCoord<1, 1>{}));
        datum(st::Pos{}, st::Z{}) += static_cast<float>(datum(llama::DatumCoord<2>{}));

        // It is also possible to work only on a part of data.
        datum(st::Pos{}) += datum(st::Momentum{});
    }
    double sum = 0.0;
    LLAMA_INDEPENDENT_DATA
    for (size_t x = 0; x < udSize[0]; ++x)
        LLAMA_INDEPENDENT_DATA
    for (size_t y = 0; y < udSize[1]; ++y)
        sum += view(x, y)(llama::DatumCoord<1, 0>{});
    std::cout << "Sum: " << sum << '\n';

    return 0;
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

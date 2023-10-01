// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <boost/iostreams/device/mapped_file.hpp>
#include <iostream>
#include <llama/llama.hpp>

// clang-format off
namespace tag
{
    struct X{} x;
    struct Y{} y;
    struct Z{} z;

    struct Normal{} normal;
    struct A{} a;
    struct B{} b;
    struct C{} c;
    struct AttribCount{} attribCount;
} // namespace tag

using Vertex = llama::Record<
    llama::Field<tag::X, float>,
    llama::Field<tag::Y, float>,
    llama::Field<tag::Z, float>
>;

using Triangle = llama::Record<
    llama::Field<tag::Normal, Vertex>,
    llama::Field<tag::A, Vertex>,
    llama::Field<tag::B, Vertex>,
    llama::Field<tag::C, Vertex>,
    llama::Field<tag::AttribCount, std::uint16_t>
>;
// clang-format on

template<typename View>
auto computeCentroid(const View& triangles)
{
    llama::One<Vertex> centroid{};
    for(const auto& t : triangles)
        centroid += t(tag::a) + t(tag::b) + t(tag::c);
    return centroid / triangles.extents()[0] / 3;
}

auto main(int argc, const char* argv[]) -> int
{
    if(argc != 2)
    {
        std::cerr << "Please pass the location of teapot.stl as argument. It is found inside: "
                     "<gitrepo>/examples/memmap/teapot.stl\n";
        return 1;
    }

    // memory map the teapot geometry file
    auto file = boost::iostreams::mapped_file_source(argv[1]);
    const auto* content = reinterpret_cast<const std::byte*>(file.data());
    const auto size = file.size();

    // binary STL header is 80 bytes, followed by uint32 triangle count, followed by vertex data without padding
    const auto n = *reinterpret_cast<const std::uint32_t*>(content + 80);
    const auto mapping = llama::mapping::AoS<
        llama::ArrayExtents<uint32_t, llama::dyn>,
        Triangle,
        llama::mapping::FieldAlignment::Pack,
        llama::mapping::LinearizeArrayIndexRight,
        llama::mapping::PermuteFieldsInOrder>{{n}};
    if(size != 80u + 4u + mapping.blobSize(0))
    {
        std::cout << "File size (" << size << ") != 80 + 4 + mapping size: (" << mapping.blobSize(0) << ")\n";
        return 1;
    }

    // create a LLAMA view on the memory mapped file (no copy is performed!)
    const auto triangles = llama::View{mapping, llama::Array{content + 84}};

    std::cout << "First triangle: " << triangles[0] << "\n";

    const auto centroid = computeCentroid(triangles);
    std::cout << "Teapot center: " << centroid << "\n";

    return 0;
}

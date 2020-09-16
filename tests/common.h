#pragma once

#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/core/demangle.hpp>
#include <llama/Allocators.hpp>
#include <numeric>
#include <regex>
#include <string>
#include <typeinfo>

template <typename T>
std::string prettyPrintType(const T& t) {
    auto raw = boost::core::demangle(typeid(t).name());
#ifdef _MSC_VER
    // remove clutter in MSVC
    boost::replace_all(raw, "struct ", "");
#endif
#ifdef __GNUG__
    // remove clutter in g++
    static std::regex ulLiteral{"(\\d+)ul"};
    raw = std::regex_replace(raw, ulLiteral, "$1");
#endif

    boost::replace_all(raw, "<", "<\n");
#ifdef _MSC_VER
    boost::replace_all(raw, ",", ",\n");
#else
    boost::replace_all(raw, ", ", ",\n");
#endif
    boost::replace_all(raw, " >", ">");
    boost::replace_all(raw, ">", "\n>");

    std::vector<std::string> tokens;
    boost::algorithm::split(tokens, raw, [](char c) { return c == '\n'; });
    std::string result;
    int indent = 0;
    for (auto t : tokens) {
        if (t.back() == '>' ||
            (t.length() > 1 && t[t.length() - 2] == '>'))
            indent -= 4;
        for (int i = 0; i < indent; ++i)
            result += ' ';
        result += t + "\n";
        if (t.back() == '<')
            indent += 4;
    }
    if (result.back() == '\n')
        result.pop_back();
    return result;
}

namespace internal
{
    inline void zeroBlob(std::shared_ptr<std::byte[]> & sp, size_t blobSize)
    {
        std::memset(sp.get(), 0, blobSize);
    }
    template<typename A>
    void zeroBlob(std::vector<std::byte, A> & v, size_t blobSize)
    {
        std::memset(v.data(), 0, blobSize);
    }
}

template<typename View>
void zeroStorage(View & view)
{
    for(auto i = 0; i < View::Mapping::blobCount; i++)
        internal::zeroBlob(view.storageBlobs[i], view.mapping.getBlobSize(i));
}

template <typename View>
void iotaStorage(View& view) {
    for (auto i = 0; i < View::Mapping::blobCount; i++) {
        auto fillFunc = [val = 0]() mutable { return static_cast<typename View::BlobType::PrimType>(val++); };
        std::generate_n(view.storageBlobs[i].storageBlobs.get(), view.mapping.getBlobSize(i), fillFunc);
    }
}

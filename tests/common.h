#pragma once

#include <boost/algorithm/string.hpp>
#include <boost/core/demangle.hpp>
#include <string>

template <typename T>
std::string prettyPrintType(const T& t) {
    auto raw = boost::core::demangle(typeid(t).name());
    boost::replace_all(raw, "<", "<\n");
    boost::replace_all(raw, ", ", ",\n");
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

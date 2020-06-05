#pragma once

#include <llama/llama.hpp>
#include <string>

template <typename T>
std::string type(const T& t) {
    return llama::demangleType(typeid(t).name());
}

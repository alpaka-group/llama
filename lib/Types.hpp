#pragma once

#include "Array.hpp"

namespace llama
{

template <size_t dim>
using UserDomain = Array<size_t,dim>;

template <typename... Leaves>
struct DateStruct;

} //namespace llama

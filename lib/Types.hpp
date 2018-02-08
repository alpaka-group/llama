#pragma once

#include "Array.hpp"

namespace llama
{

template <size_t dim>
using UserDomain = Array<size_t,dim>;

struct BlobAdress
{
	size_t blobNr;
	size_t bytePos;
};

template <typename... Leaves>
struct DateStruct;

} //namespace llama

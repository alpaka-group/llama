/* Copyright 2018 Alexander Matthes
 *
 * This file is part of LLAMA.
 *
 * LLAMA is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * LLAMA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with LLAMA.  If not, see <www.gnu.org/licenses/>.
 */

#pragma once

/**
 * \mainpage LLAMA API documentation
 *
 * LLAMA is a C++17 template header-only library for the abstraction of memory
 * access patterns. It distinguishes between the view of the algorithm on the
 * memory and the real layout in the background. This enables performance
 * portability for multicore, manycore and gpu applications with the very same
 * code.
 *
 * In contrast to many other solutions LLAMA can define nested data structures
 * of arbitrary depths and is not limited only to struct of array and array of
 * struct data layouts but is also capable to explicitly define padding,
 * blocking, striding and any other run time or compile time access pattern
 * simultaneously.
 *
 * To archieve this goal LLAMA is splitted in mostly independent, orthogonal
 * parts completely written in modern C++17 to run on as many architectures and
 * with as many compilers as possible while still supporting extensions needed
 * e.g. to run on GPU or other many core hardware.
 *
 * This page documents the API of LLAMA. The user documentation and an overview
 * about the concepts and ideas can be found here: https://llama-doc.rtfd.io
 *
 * LLAMA is licensed under the LGPL2+.
 */

#define LLAMA_VERSION_MAJOR 0
#define LLAMA_VERSION_MINOR 1
#define LLAMA_VERSION_PATCH 0

#include "DatumStruct.hpp"
#include "Factory.hpp"
#include "ForEach.hpp"
#include "Types.hpp"
#include "UserDomain.hpp"
#include "VirtualView.hpp"
#include "allocator/SharedPtr.hpp"
#include "allocator/Stack.hpp"
#include "allocator/Vector.hpp"
#include "mapping/AoS.hpp"
#include "mapping/One.hpp"
#include "mapping/SoA.hpp"
#include "mapping/tree/Mapping.hpp"
#include "preprocessor/macros.hpp"

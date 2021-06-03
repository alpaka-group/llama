# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.3] - 2021-06-04

### Added features
- added multi-blob SoA mapping allowing to map to one blob per field #111
- added `llama::FlatRecordDim` and `llama::flatRecordCoord` to flatten record dimension and record coordinates #241
- added `llama::VirtualRecord::asTuple` and `llama::VirtualRecord::asFlatTuple` to create tuples of references from virtual records #139, #141
- added an iterator for `llama::View` and `llama::View::begin`/`llama::View::end`, allowing it to be used with the STL #158, #162, #207, #259, #259
- added support for arrays of static size inside `llama::Field` #164
- added `llama::mapping::maxLanes` to help building AoSoA mappings #181
- added `llama::flatSizeOf` and `llama::flatOffetOf` working on type lists #241
- added `llama::fieldCount<RecordDim>` #241
- added `llama::LeafRecordCoords` creating a type list of record coordinates of the leaf fields #254
- added literal operator `_RC` to easy creating of record coordinates #144
- added concepts `llama::BlobAllocator` and `llama::StorageBlob` #145, #146
- added new `Heatmap` mapping, tracking bytewise memory access #192
- added `llama::forEachADCoord` to iterate over array dimensions #198
- added a parameter to AoS to support alignment #156
- made ArrayDomainIndexIterator and ArrayDomainIndexRange constexpr #130
- added support for structured bindings on `llama::VirtualRecord` #142
- added load and store support between virtual datum and any tuple like type #143
- added prototype of computed properties (experimental and undocumented) #171
- added `LLAMA_LAMBDA_INLINE` to force inlining of lambda functions #264
- added CUDA n-body #129, #132, #220, #221
- extended n-body, vectoradd and heatequation examples with more variants #115, #116, #118, #124, #133, #134, #135, #207, #213, #216, #270, #273
- added new bufferguard example #166
- added new viewcopy example, comparing various approaches to copy between `llama::View`s #119, #120, #223, #224, #25, #228, #235, #247, #268
- added new alpaka nbody example using Vc #128
- added icpc, icpx and clang to CI #157, #172
- added .clang-tidy file #195
- added clang-format check to CI #127
- extended `llama::sizeOf` and `llama::offsetOf` to support alignment and padding #156
- `llama::ArrayDomainIndexIterator` is now random access and supports C++20 ranges #199
- `llama::structName` can now be used with a type argument as well #241
- `llama::One` can be constructed from other virtual records #256
- added `llama::AlignedAllocator`
- made `llama::forEachLeaf` constexpr
- made all mappings constexpr

### Breaking changes
- renamed datum domain to record dimension, including corresponding files, helper functions, variables etc. #194, notably:
  * renamed `llama::VirtualDatum` to `llama::VirtualRecord`
  * renamed `llama::DatumCoord` to `llama::RecordCoord`
  * renamed `llama::DatumStruct` to `llama::Record`
  * renamed `llama::DatumElement` to `llama::Field`
- replaced `llama::allocVirtualDatumStack` by `llama::One` #140
- bumped required alpaka version in examples to 0.7 and adapt to changes in alpaka #113
- bumped required CMake version to 3.16 #122
- added `arrayDims` getter to all mappings and made `ArrayDims` member private #210
- renamed `llama::mapping::SplitMapping` to `llama::mapping::Split` #155
- renamed namespace `llama::allocator` to `llama::bloballoc` #188
- renamed `getBlobSize`/`getBlobNrAndOffset` in all mappings to `blobSize`/`blobNrAndOffset` #191
- removed unnecessary size argument of `llama::VirtualView` constructor
- replaced parallel STL by OpenMP in examples to remove dependency on TBB #198
- switched to clang-format 12 #202
- reorganized internal LLAMA headers #123
- `llama::offsetOf` now requires a `RecordCoord` instead of integral indices
- bumped required Boost version to 1.70

### Bug fixes and improvements
- added a few missing asymmetric arithmetic and relational operators to `llama::VirtualRecord` #115
- fixed blob splitting in `llama::mapping::Split` #155
- only write back velocity in n-body example #249
- improved output of dumped mapping visualizations #154, #265
- improved and expanded documentation and add new figures
- improved compilation time #241, #246, #254
- improve annotation of llama functions with LLAMA_FN_HOST_ACC_INLINE #152
- removed some dependencies on Boost #161, #204, #266
- updated .zenodo.json #121
- fix wrong distance calculation in body example
- refactored common timing functions in examples into class `Stopwatch`
- CMakeLists.txt cleanup
- refactored internals

### Removed features
- removed `llama::Index<I>`, use `llama::RecordCoord` now #144
- removed `llama::DatumArray` and `llama::DA`, replaced by using C++ native fixed size arrays #164

## [0.2] - 2020-10-07

- C++17 and CUDA 11
- MSVC support
- improved API using C++17 CTAD
- improved integration with Alpaka
- dump mapping visualizations
- add experimental Trace and Split meta mappings
- lots of refactoring and code improvements
- greatly updated documentation
- turn some examples into proper unit tests
- add more unit tests
- CI support with unit tests, address sanitizer, amalgamated llama.hpp, doxygen etc.
- replace png++ by stb_image
- added .clang-format file


## [0.1] - 2018-10-19

Basic functionality implemented

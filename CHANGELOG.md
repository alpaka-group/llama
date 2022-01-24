# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


## [0.4] - 2022-01-25

### Added features
- added `operator<<` for `llama::VirtualRecord`, `llama::RecordCoord`, `llama::Array` and `llama::ArrayExtents` #279, #243, #373, #374
- allow to use static arrays as record dimension #285, #244
- added `llama::copy` for layout aware copying between two views #289
- added `llama::Vector` as analog to `std::vector`, but supports LLAMA mappings #296, #300
- added CI tests for MacOS 10.15 and 11 #297, #306, #393
- added `push_front`, `pop_front`, `push_back` and `pop_back` for `llama::Array` #307
- added `operator==` and `operator!=` for `llama::RecordCoord` #308
- support arbitrary many record coords in `llama::Cat` and `llama::cat` #308
- added example showing a particle-in-cell (PIC) simulation #319
- `llama::Array` now has a member function `size` #325
- added `llama::isComputed<Mapping, RecordCoord>` to query whether a field is computed by a mapping #325
- added `llama::swap` for `VirtualRecord`, used by STL algorithms #344
- extended blob allocators to allow requesting blob alignment, now used by `llama::allocView` #339, #355
- added `llama::alignOf` and `llama::flatAlignOf` #355
- added traits to detect whether a type is a certain LLAMA mapping #359, #456
- added `TransformLeaves<RecordDim, TypeFunctor>` meta function #365
- added macros `LLAMA_FORCE_INLINE` and `LLAMA_HOST_ACC` #366
- support clang as CUDA compiler #366
- `llama::mapping::SoA` and `llama::mapping::AoSoA` now support custom record dimension flatteners #371
- added the `llama::mapping::FlattenRecordDimIncreasingAlignment`, `llama::mapping::FlattenRecordDimDecreasingAlignment` and `llama::mapping::FlattenRecordDimMinimizePadding` record dimension flatteners #371
- added new mapping `llama::mapping::BitPackedIntSoA` bitpacking integers in the record dimension into SoA arrays, and added new example #372, #427, #441, #446
- added new mapping `llama::mapping::BitPackedFloatSoA` bitpacking floating-point types in the record dimension into SoA arrays, and added new example #414, #427, #446
- `LLAMA_FORCE_INLINE` views can be created on `const` blobs #375
- added `llama::allocViewUninitialized` to create a `llama::View` without running the field type's constructors #377
- added `llama::constructFields` to run the constructors of all field type's in a view #377
- LLAMA's unit tests can now be run from the `ctest` test driver (not recommended because slower) #384
- added support for compile time array dimensions with new classes `llama::ArrayExtents` #391
- allow suppressing console output from `llama::mapping::Trace` on destruction #391
- added new mapping `llama::mapping::Bytesplit` that allows to split each field type into a byte array and map using a further mapping, and added example #395, #398, #399, #441
- added macro `LLAMA_UNROLL` to request unrolling of a loop #403
- allow `llama::VirtualView` to store its inner view #406
- `llama::mapping::Split` now supports multiple record coords to select how the record dimension is split #407
- added clang-12, clang-13, g\++-9, g++-11, nvcc 11.3, 11.4, 11.5, 11.6, Visual Studio 2022 to CI #314, #315, #317, #335, #396, #408, #412, #461
- added `CopyConst` type function #419
- added new mapping `llama::mapping::ChangeType` that replaces types from the record dimension for other types when storing #421, #441
- added new mixin `llama::ProxyRefOpMixin` to help supporting compount assignment and increment/decrement operators on proxy references #430
- added unit test coverage analysis and reports for each PR #432
- added new `llama::mapping::Null` mapping, that maps elements to nothing, discarding written values and returning default constructed values when reading #442
- added new example `daxpy` focusing on the mappings `llama::mapping::BitPackedFloatSoA`, `llama::mapping::Bytesplit` and `llama::mapping::ChangeType` #450, #452, #455
- added `llama::ReplacePlaceholders` meta function #451

### Breaking changes
- develop is the new default branch on GitHub, master was deleted #280
- `llama::One` is now a zero-dimensional view (instead of one-dimensional) #286
- `llama::mapping::AoS` is aligned and `llama::mapping::SoA` is multiblob by default #312
- all alpaka examples now require alpaka 0.7 #321
- updated clang-format to version 12.0.1 #326, #404
- stricter checking whether a type is allowed as field type in general #284
- stricter checking whether a type is allowed as field type in `llama::copy` #329
- `llama::allocView` will now execute the constructors of the field type's #377
- brightened the colors used for dumped mapping visualizations #387
- renamed `llama::forEachLeaf` to `llama::forEachLeafCoord` and added new `llama::forEachLeaf` iterating over the fields of a record #388
- replaced `llama::ArrayDims` by `llama::ArrayExtents` and `llama::ArrayIndex` #391
- renamed `llama::ArrayDimsIndexIterator` to `llama::ArrayIndexIterator` #391
- renamed `llama::ArrayDimsIndexRange` to `llama::ArrayIndexRange` #391
- renamed `llama::mapping::Mapping::arrayDims()` -to `llama::mapping::Mapping::extents()` #391
- the `ASAN_FOR_TESTS` CMake option has been renamed to `LLAMA_ENABLE_ASAN_FOR_TESTS` #425
- renamed all `llama::mapping::PreconfiguredMapping` meta functions to `llama::mapping::BindMapping` #456

### Bug fixes and improvements
- updated zenodo file and provided a DOI to LLAMA's releases #282, #291, #292
- views can be indexed with signed integer types as well #283
- improve `LLAMA_INDEPENDENT_DATA` for clang compilers and the Intel LLVM compiler (icx) #302, #411
- fixed a missing include #304
- made `llama::Tuple` more similar to `std::tuple` #309
- added clang-tidy CI checks #310, #367
- all CMake projects now only request C++ as language #321
- `llama::One` now respects the field type's alignment and minimizes its size #323
- fixed `LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS` for nvcc when using MSVC as host compiler #334
- fixed AoSoA blob size when the flat array extent is not divisible by the `Lanes` parameter #336
- switched MSVC C++ standard flag from `/std:c++20` to `/std:c++latest` for unit tests #338, #443
- added more unit tests for `std::transform` on LLAMA views #343
- fixed `value_type` of `View::iterator` to be STL compatible #346
- fixed default arguments for `llama::mapping::PreconfiguredAoS` to match `llama::mapping::AoS` #347
- fixed default arguments for `llama::mapping::PreconfiguredSoA` to match `llama::mapping::SoA` #369
- improved `llama::VirtualRecord`'s and `llama::View`'s size using empty base optimization #348
- updated `stb` third-party libraries #352
- ensured proper truncation of empty space after `hostname()` in common example utilities #353
- a mapping's `blobNrAndOffset` can now deduce the record coordinates from a passed instance of `llama::RecordCoord` #368
- provided `boost::mp11::mp_flatten` if Boost version is too old #370
- ensured that `llama::VirtualView` supports negative indices #379
- documented the behavior of the array extents linearizers #380
- the fmt library is now an optional dependency for the llama CMake target #382, #383
- the unit tests now compile with higher warning levels #386
- better checking for unnecessary `const` qualifiers on `Mapping` and `ArrayExtents` template arguments #391
- refactoring CMake optimization flags #392
- refactored unit tests #299, #397
- added more unit tests for `llama::bloballoc::AlignedAllocator` and `llama::mapping::Trace` #437
- fixed generating invalid CSS class names for HTML dumps #410
- avoid blurry heatmaps dumped by `llama::mapping::Heatmap` #416
- ensure that a fully-static `llama::ArrayExtents` and `llama::mapping::One` are stateless #417
- `DumpMapping.hpp` is now included via `llama.hpp` (with disabled content when the fmt library is not available) #251, #422
- added `Bytesplit` and `BitpackedFloatSoA` mappings to n-body and heatequation examples #431
- simplified implementation of `llama::tupleReplace` #435
- `llama::Tuple` does no longer reserve space for empty types #436
- improved documentation and README.md #440, #445, #453, #454, #457
- fixed detection whether compilers support C++20 ranges #443
- refined mapping related concepts #444
- CI switched to Boost 1.74 because of alpaka
- support templates in `llama::structName` #449

### Removed features

- dropped support for the Intel C++ Compiler Classic (icp) #351
- removed `llama::Array::rank`, replaced by `llama::Array::size` #391

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

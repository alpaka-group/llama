# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).


## [0.5] - 2023-01-31

### Added features
- allow record coords in `llama::mapping::ChangeType`'s replacement map #468
- converted the daxpy example to alpaka, so it can be used on more architectures #469
- added new CUDA demo for pitched allocation #473
- added small utilities `llama::divCeil`, `llama::roundToMultiple` and `llama::dot(Array)` #477
- added support for new compilers/OSes: clang-14 #484, clang-15 #590, gcc-12 #490, nvcc 11.7 #501, nvcc 11.8 #591, nvcc 12.0 #654, MacOS-12 #540, nvc++ 22.9 (nvhpc) #547, #589
- support array extents with arbitrary value types #488
- the creation of the single amalgamated header is now available as script: #497, #535
- a single amalgamated header from LLAMA is now published on each commit: #535
- the `Trace` mapping is now supported on GPUs #503
- the `Heatmap` mapping is now supported on GPUs #587
- added macros for likely and unlikely attributes #506
- added `front()` and `back()` to `llama::Array` #517, #528
- added `data()` to `llama::Array` #553
- allow in-place construction of `llama::Trace`'s inner mapping #517
- make printing API in `llama::Trace` more versatile #517
- added a documentation section comparing C++ and LLAMA data structure access #522
- documented interplay of member functions and proxy references #524
- added new utility functions `llama::transformBlobs()` and `llama::shallowCopy()` #525
- added `llama::isTrace` trait #529
- documented how to form references to `llama::One` #532
- documented new LLAMA mappings and accessors #545, #583, #640
- added `llama::isOne` trait #549
- added `llama::isProxyReference` trait #550
- added `llama::ScopedUpdate`, a tool to generically update values through a (proxy) reference #550
- added an API for explicit SIMD programming #577, #578, #581
- data access can now be customized using accessors #579, #611, #612, #642
- the `README.md` has been updated with a link to our first publication on LLAMA #596
- all mappings now re-expose their template parameters as nested types/values #599
- added the `Projection` and `Byteswap` mappings #607, #612
- added an example viewing a memory mapped file #608
- heatmaps can now be written to binary files in addition to ASCII #615
- added meta mapping `llama::mapping::PermuteArrayIndex` to permute array indices #616, #636
- heatmap output can be trimmed #618
- added blob allocator `llama::bloballoc::UniquePtr` #630
- added STREAM benchmark #643
- added some preliminary support for HIP (not CI tested yet) #651
- added the BabelStream benchmark #650
- added ROOT LHCB B2HHH analysis example #660, #672, #684
- the `Split` mapping now additionally supports tag lists as selectors #674
- allow the `BitPackedInt*` mappings to omit the sign bit #675
- added new mapping `BitPackedIntAoS` #678
- added new mapping `BitPackedFloatAoS` #687
- improved array handling of `recordCoordTags` #693

### Breaking changes

- the template parameter list for `llama::ArrayExtents` changed to support specifying the index type: #488
- the CI now uses alpaka 0.9 and not the development version #492
- LLAMA's cmake project now builds in Release mode by default with tests/examples off #509
- the unit tests now require Catch2 v3 to build, which can be downloaded automatically or taken from the system #511, #570
- cmake 3.18.3 is now required by LLAMA and all examples #526
- renamed `llama::VirtualRecord` into `llama::RecordRef` #551
- the `Vc` library has been replaced by `xsimd` for explicit vectorization #557
- the requirements on computed mappings have been tightened #627
- renamed blob allocator `llama::bloballoc::Stack` to `llama::bloballoc::Array` #629
- renamed `llama::VirtualView` to `llama::SubView` #638
- the `SoA` mapping now aligns subarrays by default if a single blob is used #648
- replaced Boolean parameters of mappings by enums to increase readability #655
- the `Trace` mapping has been renamed to `FieldAccessCount` #690
- replaced `.zenodo.json` by `CITATION.cff` #696
- renamed `recordCoordTags` into `prettyRecordCoord` #693

### Bug fixes and improvements
- fixed various compilation flags #470
- aligned `std::vector` in `daxpy` baseline benchmark #471
- refactored common mapping code into a shared base class #472
- fixed alpaka examples to support alpaka 0.9 #474, #504
- made Codecov reports on PRs less verbose and allow for small coverage decreases #475
- removed some MSVC workarounds #476
- various minor CI fixes and updates: #478, #479, #483, #485, #491, #493, #494, #505, #512, #515, #519, #533, #538, #546, #556, #558, #562, #569, #571, #586, #600, #601, #602, #619, #620, #621, #622, #645, #646, #686, #688
- various small code fixes: #486, #489, #495, #500, #502, #507, #527, #560, #575, #584, #597, #598, #603, #617, #631, #632, #641, #649, #658, #659, #673
- various documentation fixes: #496, #514, #543, #563, #588, #624, #644, #649, #689
- various unit test improvements: #531, #534, #537, #568, #609, #613, #661, #698
- fixed empty base optimization for MSVC: #499
- `llama::structName<T>()` and `llama::recordCoordTags<T>` are now `constexpr` #521
- cmake variables from Catch are now hidden by default in cmake guis #548
- fixed warnings and asserts, and improve bitpacked mappings #549, #671, #677, #681
- fixed some edge cases and improved mapping dumping #552, #647
- allow assigning Trace references directly to each other #555
- the naming of identifiers in LLAMA code is now enforced by `clang-tidy` #565
- code formatting now requires `clang-format-15` #508, #564, #685
- support proxy references in RecordRef tuple interface #572
- comply to new CRP clang-tidy checks #573
- the runs of the n-body example are now verified against each other #574
- suppress unnecessary CUDA warnings #580
- the n-body and alpaka n-body example are now more similar and support explicit SIMD #582
- the gnuplot scripts for heatmaps have been improved #623
- a view constructed without a blob array argument will now value initialize the blob array #649
- the `SoA` mapping's performance has been improved when the array extents are fully known at compile time #653
- fix `llama::structName<T>()` for `T`s in unnamed namespaces

### Removed features

- support for Visual Studio 2019 has been dropped #539
- support for MacOS 11.15 has been dropped #561
- support for AppleClang has been dropped, use brew's clang on MacOS #593
- the obsolete `nbody_benchmark` example has been removed #595

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

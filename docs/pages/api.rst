.. include:: common.rst

API
===

Users should just include :cpp:`llama.hpp` and all functionality should be available.
All basic functionality of the library is in the namespace :cpp:`llama` or sub namespaces.

Useful helpers
--------------

.. _label-api-array:

.. doxygenstruct:: llama::NrAndOffset
   :members:
.. doxygenfunction:: llama::structName(T)
.. doxygentypedef:: CopyConst
.. doxygenstruct:: llama::ProxyRefOpMixin
   :members:

Array
^^^^^

.. doxygenstruct:: llama::Array
   :members:
.. doxygenfunction:: llama::pushFront
.. doxygenfunction:: llama::pushBack
.. doxygenfunction:: llama::popFront([[maybe_unused]] Array<T, N> a)
.. doxygenfunction:: llama::popBack
.. doxygenfunction:: llama::product(Array<T, N> a)

Tuple
^^^^^

.. doxygenstruct:: llama::Tuple
   :members:
.. doxygenfunction:: llama::get(Tuple<Elements...> &tuple)
.. doxygenfunction:: llama::tupleCat
.. doxygenfunction:: llama::tupleReplace
.. doxygenfunction:: llama::tupleTransform
.. doxygenfunction:: llama::popFront(const Tuple<Elements...> &tuple)

Array dimensions
----------------

.. doxygenstruct:: llama::ArrayExtents
.. doxygentypedef:: llama::ArrayExtentsDynamic
.. doxygentypedef:: llama::ArrayExtentsNCube
.. doxygenstruct:: llama::ArrayIndex

.. doxygenstruct:: llama::ArrayIndexIterator
   :members:
.. doxygenstruct:: llama::ArrayIndexRange
   :members:

.. doxygenfunction:: llama::forEachADCoord(ArrayExtents<SizeType, Sizes...> extents, Func&& func)

Record dimension
----------------

.. doxygenstruct:: llama::Record
.. doxygenstruct:: llama::Field
.. doxygenstruct:: llama::NoName

.. doxygentypedef:: llama::GetFieldTag
.. doxygentypedef:: llama::GetFieldType
.. doxygenvariable:: llama::offsetOf
.. doxygenvariable:: llama::sizeOf
.. doxygenvariable:: llama::alignOf
.. doxygentypedef:: llama::GetTags
.. doxygentypedef:: llama::GetTag
.. doxygenvariable:: llama::hasSameTags
.. doxygentypedef:: llama::GetCoordFromTags
.. doxygentypedef:: llama::GetType
.. doxygentypedef:: llama::FlatRecordDim
.. doxygenvariable:: llama::flatRecordCoord
.. doxygentypedef:: llama::LeafRecordCoords
.. doxygentypedef:: llama::TransformLeaves
.. doxygentypedef:: llama::MergedRecordDims

.. doxygenfunction:: llama::forEachLeafCoord(Functor &&functor, Tags... baseTags)
.. doxygenfunction:: llama::forEachLeafCoord(Functor &&functor, RecordCoord<Coords...> baseCoord)

.. doxygenfunction:: llama::prettyRecordCoord

Record coordinates
------------------

.. doxygenstruct:: llama::RecordCoord
   :members:
.. doxygentypedef:: llama::RecordCoordFromList
.. doxygentypedef:: llama::Cat
.. doxygentypedef:: llama::PopFront
.. doxygenvariable:: llama::recordCoordCommonPrefixIsBigger
.. doxygenvariable:: llama::recordCoordCommonPrefixIsSame

Views
-----

.. _label-api-allocView:
.. doxygenfunction:: llama::allocView
.. doxygenfunction:: llama::constructFields
.. doxygenfunction:: llama::allocViewUninitialized
.. doxygenfunction:: llama::allocViewStack
.. doxygentypedef:: llama::One
.. doxygenfunction:: llama::copyRecord

.. doxygenfunction:: transformBlobs
.. doxygenfunction:: shallowCopy

.. doxygenfunction:: withMapping
.. doxygenfunction:: withAccessor

.. _label-api-bloballocators:

Blob allocators
^^^^^^^^^^^^^^^

.. doxygenstruct:: llama::bloballoc::Vector
   :members:
.. doxygenstruct:: llama::bloballoc::SharedPtr
   :members:
.. doxygenstruct:: llama::bloballoc::UniquePtr
   :members:
.. doxygenstruct:: llama::bloballoc::Array
   :members:

Mappings
--------

.. doxygenstruct:: llama::mapping::AoS
   :members:
.. doxygentypedef:: llama::mapping::AlignedAoS
.. doxygentypedef:: llama::mapping::MinAlignedAoS
.. doxygentypedef:: llama::mapping::PackedAoS
.. doxygentypedef:: llama::mapping::SingleBlobSoA
.. doxygentypedef:: llama::mapping::MultiBlobSoA
.. doxygenstruct:: llama::mapping::AoSoA
   :members:
.. doxygenvariable:: llama::mapping::maxLanes
.. doxygenstruct:: llama::mapping::BitPackedIntAoS
   :members:
.. doxygenstruct:: llama::mapping::BitPackedIntSoA
   :members:
.. doxygenstruct:: llama::mapping::BitPackedFloatAoS
   :members:
.. doxygenstruct:: llama::mapping::BitPackedFloatSoA
   :members:
.. doxygenstruct:: llama::mapping::Bytesplit
   :members:
.. doxygenstruct:: llama::mapping::Byteswap
   :members:
.. doxygenstruct:: llama::mapping::ChangeType
   :members:
.. doxygenstruct:: llama::mapping::Heatmap
   :members:
.. doxygenstruct:: llama::mapping::Null
   :members:
.. doxygenstruct:: llama::mapping::One
   :members:
.. doxygenstruct:: llama::mapping::Projection
   :members:
.. doxygenstruct:: llama::mapping::SoA
   :members:
.. doxygenstruct:: llama::mapping::Split
   :members:
.. doxygenstruct:: llama::mapping::FieldAccessCount
   :members:

Acessors
^^^^^^^^

.. doxygenstruct:: llama::accessor::Default
.. doxygenstruct:: llama::accessor::ByValue
.. doxygenstruct:: llama::accessor::Const
.. doxygenstruct:: llama::accessor::Restrict
.. doxygenstruct:: llama::accessor::Atomic

RecordDim flattener
^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: llama::mapping::FlattenRecordDimInOrder
.. doxygenstruct:: llama::mapping::FlattenRecordDimSorted
.. doxygentypedef:: llama::mapping::FlattenRecordDimIncreasingAlignment
.. doxygentypedef:: llama::mapping::FlattenRecordDimDecreasingAlignment
.. doxygentypedef:: llama::mapping::FlattenRecordDimMinimizePadding

Common utilities
^^^^^^^^^^^^^^^^

.. doxygenstruct:: llama::mapping::LinearizeArrayDimsCpp
   :members:
.. doxygenstruct:: llama::mapping::LinearizeArrayDimsFortran
   :members:
.. doxygenstruct:: llama::mapping::LinearizeArrayDimsMorton
   :members:

Tree mapping (deprecated)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: llama::mapping::tree::Mapping
   :members:

For a detailed description of the tree mapping concept have a look at
:ref:`LLAMA tree mapping <label-tree-mapping>`

**Tree mapping functors**

.. doxygenstruct:: llama::mapping::tree::functor::Idem
.. doxygenstruct:: llama::mapping::tree::functor::LeafOnlyRT
.. doxygenstruct:: llama::mapping::tree::functor::MoveRTDown

.. FIXME: doxygen fails to parse the source code ...

Dumping
^^^^^^^

.. doxygenfunction:: llama::toSvg
.. doxygenfunction:: llama::toHtml

Data access
-----------

.. doxygenstruct:: llama::View
   :members:
.. doxygenstruct:: llama::SubView
   :members:
.. doxygenstruct:: llama::RecordRef
   :members:

Copying
-------

.. doxygenfunction:: llama::copy
.. doxygenstruct:: llama::Copy
   :members:
.. doxygenfunction:: llama::fieldWiseCopy
.. doxygenfunction:: llama::aosoaCommonBlockCopy

SIMD
----

.. doxygenstruct:: llama::SimdTraits
.. doxygenvariable:: llama::simdLanes
.. doxygentypedef:: llama::SimdizeN
.. doxygentypedef:: llama::Simdize
.. doxygenvariable:: llama::simdLanesFor
.. doxygentypedef:: llama::SimdN
.. doxygentypedef:: llama::Simd
.. doxygenfunction:: llama::loadSimd
.. doxygenfunction:: llama::storeSimd
.. doxygenfunction:: llama::simdForEachN
.. doxygenfunction:: llama::simdForEach

Macros
------

.. doxygendefine:: LLAMA_INDEPENDENT_DATA
.. doxygendefine:: LLAMA_FORCE_INLINE
.. doxygendefine:: LLAMA_UNROLL
.. doxygendefine:: LLAMA_HOST_ACC
.. doxygendefine:: LLAMA_FN_HOST_ACC_INLINE
.. doxygendefine:: LLAMA_LAMBDA_INLINE
.. doxygendefine:: LLAMA_COPY

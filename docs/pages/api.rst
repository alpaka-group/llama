.. include:: common.rst

API
===

Users should just include :cpp:`llama.hpp` and all functionality should be available.
All basic functionality of the library is in the namespace :cpp:`llama` or sub namespaces.

Useful helpers
--------------

.. _label-api-array:

.. doxygenstruct:: llama::Array
   :members:
.. doxygenstruct:: llama::Tuple
   :members:
.. doxygentypedef:: llama::TupleElement
.. doxygenfunction:: llama::get
.. doxygenvariable:: llama::tupleSize
.. doxygenfunction:: llama::tupleCat
.. doxygenfunction:: llama::tupleReplace
.. doxygenfunction:: llama::tupleTransform
.. doxygenfunction:: llama::tupleWithoutFirst(const Tuple<Elements...> &tuple)
.. doxygenstruct:: llama::NrAndOffset
   :members:
.. doxygenfunction:: llama::structName

Array dimensions
----------------

.. doxygenstruct:: llama::ArrayDims

.. doxygenstruct:: llama::ArrayDimsIndexIterator
   :members:
.. doxygenstruct:: llama::ArrayDimsIndexRange
   :members:

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
.. doxygentypedef:: llama::GetCoordFromTagsRelative
.. doxygentypedef:: llama::FlatRecordDim
.. doxygenvariable:: llama::flatRecordCoord
.. doxygentypedef:: llama::LeafRecordCoords

.. doxygenfunction:: llama::forEachLeaf(Functor &&functor, Tags... baseTags)
.. doxygenfunction:: llama::forEachLeaf(Functor &&functor, RecordCoord<Coords...> baseCoord)

Record coordinates
------------------

.. doxygenstruct:: llama::RecordCoord
   :members:
.. doxygentypedef:: llama::RecordCoordFromList
.. doxygentypedef:: llama::Cat
.. doxygentypedef:: llama::PopFront
.. doxygenvariable:: llama::RecordCoordCommonPrefixIsBigger
.. doxygenvariable:: llama::RecordCoordCommonPrefixIsSame

View creation
-------------

.. _label-api-allocView:
.. doxygenfunction:: llama::allocView
.. doxygenfunction:: llama::allocViewStack
.. doxygentypedef:: llama::One
.. doxygenfunction:: llama::copyVirtualRecordStack

.. _label-api-bloballocators:

Blob allocators
^^^^^^^^^^^^^^^

.. doxygenstruct:: llama::bloballoc::Vector
   :members:
.. doxygenstruct:: llama::bloballoc::SharedPtr
   :members:
.. doxygenstruct:: llama::bloballoc::Stack
   :members:

Mappings
--------

.. doxygentypedef:: llama::mapping::AlignedAoS
.. doxygentypedef:: llama::mapping::PackedAoS
.. doxygenstruct:: llama::mapping::AoS
   :members:
.. doxygentypedef:: llama::mapping::SingleBlobSoA
.. doxygentypedef:: llama::mapping::MultiBlobSoA
.. doxygenstruct:: llama::mapping::SoA
   :members:
.. doxygenstruct:: llama::mapping::One
   :members:
.. doxygenstruct:: llama::mapping::AoSoA
   :members:
.. doxygenvariable:: llama::mapping::maxLanes
.. doxygenstruct:: llama::mapping::Split
   :members:
.. doxygenstruct:: llama::mapping::Trace
   :members:
.. doxygenstruct:: llama::mapping::Heatmap
   :members:

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
.. doxygenstruct:: llama::VirtualView
   :members:
.. doxygenstruct:: llama::VirtualRecord
   :members:

Copying
-------

.. doxygenfunction:: llama::copy
.. doxygenstruct:: llama::Copy
   :members:
.. doxygenfunction:: llama::fieldWiseCopy
.. doxygenfunction:: llama::aosoaCommonBlockCopy

Macros
------

.. doxygendefine:: LLAMA_INDEPENDENT_DATA
.. doxygendefine:: LLAMA_FORCE_INLINE
.. doxygendefine:: LLAMA_HOST_ACC
.. doxygendefine:: LLAMA_FN_HOST_ACC_INLINE
.. doxygendefine:: LLAMA_LAMBDA_INLINE
.. doxygendefine:: LLAMA_FORCE_INLINE_RECURSIVE
.. doxygendefine:: LLAMA_COPY

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

Array domain
-----------

.. doxygenstruct:: llama::ArrayDomain

.. doxygenstruct:: llama::ArrayDomainIndexIterator
   :members:
.. doxygenstruct:: llama::ArrayDomainIndexRange
   :members:

Datum domain
------------

.. doxygenstruct:: llama::DatumStruct
.. doxygentypedef:: llama::DS
.. doxygenstruct:: llama::DatumElement
.. doxygentypedef:: llama::DE
.. doxygentypedef:: llama::DatumArray
.. doxygentypedef:: llama::DA
.. doxygenstruct:: llama::NoName
.. doxygentypedef:: llama::Index

.. doxygentypedef:: llama::GetDatumElementTag
.. doxygentypedef:: llama::GetDatumElementType
.. doxygenvariable:: llama::offsetOf
.. doxygenvariable:: llama::sizeOf
.. doxygenvariable:: llama::isDatumStruct
.. doxygentypedef:: llama::GetTags
.. doxygentypedef:: llama::GetTag
.. doxygenvariable:: llama::hasSameTags
.. doxygentypedef:: llama::GetCoordFromTags
.. doxygentypedef:: llama::GetType
.. doxygentypedef:: llama::GetCoordFromTagsRelative

.. doxygenfunction:: llama::forEach(Functor &&functor, Tags... baseTags)
.. doxygenfunction:: llama::forEach(Functor &&functor, DatumCoord<Coords...> baseCoord)

Datum coordinates
-----------------

.. doxygenstruct:: llama::DatumCoord
   :members:
.. doxygentypedef:: llama::DatumCoordFromList
.. doxygentypedef:: llama::Cat
.. doxygentypedef:: llama::PopFront
.. doxygenvariable:: llama::DatumCoordCommonPrefixIsBigger
.. doxygenvariable:: llama::DatumCoordCommonPrefixIsSame

View creation
-------------

.. _label-api-allocView:
.. doxygenfunction:: llama::allocView
.. doxygenfunction:: llama::allocViewStack
.. doxygenfunction:: llama::allocVirtualDatumStack
.. doxygenfunction:: llama::copyVirtualDatumStack

.. _label-api-allocators:

Allocators
^^^^^^^^^^

.. doxygenstruct:: llama::allocator::Vector
   :members:
.. doxygenstruct:: llama::allocator::SharedPtr
   :members:
.. doxygenstruct:: llama::allocator::Stack
   :members:

Mappings
--------

.. doxygenstruct:: llama::mapping::AoS
   :members:
.. doxygenstruct:: llama::mapping::SoA
   :members:
.. doxygenstruct:: llama::mapping::One
   :members:
.. doxygenstruct:: llama::mapping::AoSoA
   :members:
.. doxygenstruct:: llama::mapping::Trace
   :members:

Common utilities
^^^^^^^^^^^^^^^^

.. doxygenstruct:: llama::mapping::LinearizeArrayDomainCpp
   :members:
.. doxygenstruct:: llama::mapping::LinearizeArrayDomainFortran
   :members:
.. doxygenstruct:: llama::mapping::LinearizeArrayDomainMorton
   :members:

Tree mapping
^^^^^^^^^^^^

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
.. doxygenstruct:: llama::VirtualDatum
   :members:

Macros
------

.. doxygendefine:: LLAMA_INDEPENDENT_DATA
.. doxygendefine:: LLAMA_FN_HOST_ACC_INLINE
.. doxygendefine:: LLAMA_FORCE_INLINE_RECURSIVE
.. doxygendefine:: LLAMA_COPY

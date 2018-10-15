.. include:: ../common.rst

API
===

All functions, classes and structs (except of the macros of course) are in
the namespace :cpp:`llama`.

Useful helpers
--------------

.. _label-api-array:

.. doxygenstruct:: llama::Array
   :project: LLAMA
   :members:

.. doxygenstruct:: llama::Tuple
   :project: LLAMA
   :members:

.. doxygenfunction:: llama::makeTuple
   :project: LLAMA

.. doxygenfunction:: llama::getTupleElement
   :project: LLAMA

.. doxygenfunction:: llama::getTupleElementRef
   :project: LLAMA

.. doxygentypedef:: llama::GetTupleType
	:project: LLAMA

.. doxygenstruct:: llama::SizeOfTuple
   :project: LLAMA
   :members:

.. doxygentypedef:: TupleCatType
	:project: LLAMA

.. doxygenfunction:: llama::tupleCat
   :project: LLAMA

.. doxygenfunction:: llama::tupleReplace
   :project: LLAMA

.. doxygenfunction:: llama::tupleTransform
   :project: LLAMA

.. doxygenfunction:: llama::tupleRest
   :project: LLAMA

.. doxygenstruct:: llama::IntegerSequence
   :project: LLAMA

.. doxygentypedef:: llama::MakeIntegerSequence
   :project: LLAMA

.. doxygentypedef:: llama::MakeZeroSequence
   :project: LLAMA

Domains
-------

User domain
^^^^^^^^^^^

.. doxygentypedef:: llama::UserDomain
   :project: LLAMA

.. doxygenstruct:: llama::ExtentUserDomainAdress
   :project: LLAMA
   :members:

.. doxygenstruct:: llama::LinearizeUserDomainAdress
   :project: LLAMA
   :members:

.. doxygenstruct:: llama::LinearizeUserDomainAdressLikeFortran
   :project: LLAMA
   :members:

.. doxygenfunction:: llama::userDomainZero
   :project: LLAMA

Datum domain
^^^^^^^^^^^^

.. doxygentypedef:: llama::DatumStruct
   :project: LLAMA

.. doxygentypedef:: llama::DS
   :project: LLAMA

.. doxygentypedef:: llama::DatumElement
   :project: LLAMA

.. doxygentypedef:: llama::DE
   :project: LLAMA

.. doxygentypedef:: llama::DatumArray
   :project: LLAMA

.. doxygentypedef:: llama::DA
   :project: LLAMA

.. doxygenstruct:: llama::NoName
   :project: LLAMA

.. doxygenstruct:: llama::ForEach
   :project: LLAMA
   :members:

.. doxygentypedef:: llama::GetUID
   :project: LLAMA

.. doxygenstruct:: llama::CompareUID
   :project: LLAMA
   :members:

.. doxygentypedef:: GetCoordFromUID
   :project: LLAMA

.. doxygenstruct:: llama::LinearBytePos
   :project: LLAMA
   :members:

.. doxygenstruct:: llama::SizeOf
   :project: LLAMA
   :members:

.. doxygentypedef:: llama::GetDatumElementType
   :project: LLAMA

.. doxygentypedef:: llama::GetDatumElementUID
   :project: LLAMA

.. doxygentypedef:: llama::GetType
   :project: LLAMA

.. doxygentypedef:: llama::GetTypeFromDatumCoord
   :project: LLAMA

.. doxygenstruct:: llama::DatumCoord
   :project: LLAMA
   :members:

.. doxygenstruct:: llama::DatumCoordIsBigger
   :project: LLAMA
   :members:

.. doxygenstruct:: llama::DatumCoordIsSame
   :project: LLAMA
   :members:

View creation
-------------

Factory
^^^^^^^

.. _label-api-factory:

.. doxygenstruct:: llama::Factory
   :project: LLAMA
   :members:

.. doxygentypedef:: llama::OneOnStackFactory
   :project: LLAMA

.. doxygenfunction:: llama::tempAlloc
   :project: LLAMA

.. _label-api-allocators:

Allocators
^^^^^^^^^^

All allocators are in namespace :cpp:`llama::allocator`.

.. doxygenstruct:: llama::allocator::Vector
   :project: LLAMA
   :members:

.. doxygenstruct:: llama::allocator::SharedPtr
   :project: LLAMA
   :members:

.. doxygenstruct:: llama::allocator::Stack
   :project: LLAMA
   :members:

Alpaka allocators
"""""""""""""""""

As :ref:`already stated <label-allocators-alpaka>`, the alpaka connection is not
part of LLAMA, but was considered while developing the library. Furthermore some
examples are using alpaka for parallelization on many-core devices. Therefore
the alpaka allocators will be described here, too.

.. doxygenstruct:: common::allocator::Alpaka
   :project: LLAMA
   :members:

.. doxygenstruct:: common::allocator::AlpakaMirror
   :project: LLAMA
   :members:

.. doxygenstruct:: common::allocator::AlpakaShared
   :project: LLAMA
   :members:

Mappings
^^^^^^^^

All mappings are in namespace :cpp:`llama::mapping`.

.. doxygenstruct:: llama::mapping::AoS
   :project: LLAMA
   :members:

.. doxygenstruct:: llama::mapping::SoA
   :project: LLAMA
   :members:

.. doxygenstruct:: llama::mapping::One
   :project: LLAMA
   :members:

Tree mapping
""""""""""""

The tree mapping is in the namespace :cpp:`llama::mapping::tree` and all tree
mapping functors in the namespace :cpp:`llama::mapping::tree::functor`.

.. doxygenstruct:: llama::mapping::tree::Mapping
   :project: LLAMA
   :members:

For a detailed description of the tree mapping concept have a look at
:ref:`LLAMA tree mapping <label-tree-mapping>`

**Tree mapping functors**

.. doxygenstruct:: llama::mapping::tree::functor::Idem
   :project: LLAMA

.. doxygenstruct:: llama::mapping::tree::functor::LeafOnlyRT
   :project: LLAMA

.. doxygenstruct:: llama::mapping::tree::functor::MoveRTDown
   :project: LLAMA

Data access
-----------

View
^^^^

.. doxygenstruct:: llama::View
   :project: LLAMA
   :members:

VirtualView
^^^^^^^^^^^

.. doxygenstruct:: llama::VirtualView
   :project: LLAMA
   :members:

VirtualDatum
^^^^^^^^^^^^

.. doxygenstruct:: llama::VirtualDatum
   :project: LLAMA
   :members:

Operation overloads
"""""""""""""""""""

LLAMA implements the same overload for a big amount of operations. To not to
copy and paste the same code over and over, these overloads are defined once as
C++ preprocessor macros and then instantiated for all needed operations.

.. _label-define-foreach-functor:

.. doxygendefine:: __LLAMA_DEFINE_FOREACH_FUNCTOR
   :project: LLAMA

In LLAMA this macro is extended for these combinations of :cpp:`OP` and
:cpp:`FUNCTOR`:

==== ==============
 OP     FUNCTOR
==== ==============
  =  Assigment
 +=  Addition
 -=  Subtraction
 \*= Multiplication
 /=  Division
 %=  Modulo
==== ==============

.. doxygendefine:: __LLAMA_VIRTUALDATUM_VIRTUALDATUM_OPERATOR
   :project: LLAMA

.. doxygendefine:: __LLAMA_VIRTUALDATUM_VIEW_OPERATOR
   :project: LLAMA

.. doxygendefine:: __LLAMA_VIRTUALDATUM_TYPE_OPERATOR
   :project: LLAMA

These three macros are extended for the same combinations of OP and FUNCTOR as
for :ref:`__LLAMA_DEFINE_FOREACH_FUNCTOR <label-define-foreach-functor>` with
:cpp:`REF` being :cpp:`&` and :cpp:`&&` for each combination.

.. _label-define-foreach-bool-functor:

.. doxygendefine:: __LLAMA_DEFINE_FOREACH_BOOL_FUNCTOR
   :project: LLAMA

In LLAMA this macro is extended for these combinations of :cpp:`OP` and
:cpp:`FUNCTOR`:

==== ===============
 OP     FUNCTOR
==== ===============
 ==  SameAs
 !=  Not
 <   SmallerThan
 <=  SmallerSameThan
 >   BiggerThan
 >=  BiggerSameThan
==== ===============

.. doxygendefine:: __LLAMA_VIRTUALDATUM_VIRTUALDATUM_BOOL_OPERATOR
   :project: LLAMA

.. doxygendefine:: __LLAMA_VIRTUALDATUM_VIEW_BOOL_OPERATOR
   :project: LLAMA

.. doxygendefine:: __LLAMA_VIRTUALDATUM_TYPE_BOOL_OPERATOR
   :project: LLAMA

These three macros are extended for the same combinations of OP and FUNCTOR as
for
:ref:`__LLAMA_DEFINE_FOREACH_BOOL_FUNCTOR <label-define-foreach-bool-functor>`
with :cpp:`REF` being :cpp:`&` and :cpp:`&&` for each combination.


Parallelization helpers & Macros
--------------------------------

.. doxygendefine:: LLAMA_INDEPENDENT_DATA
   :project: LLAMA

.. doxygendefine:: LLAMA_FN_HOST_ACC_INLINE
   :project: LLAMA

.. doxygendefine:: LLAMA_NO_HOST_ACC_WARNING
   :project: LLAMA

.. doxygendefine:: LLAMA_FORCE_INLINE_RECURSIVE
   :project: LLAMA

.. doxygendefine:: LLAMA_IF_DEBUG
   :project: LLAMA

.. doxygendefine:: LLAMA_IF_RELEASE
   :project: LLAMA

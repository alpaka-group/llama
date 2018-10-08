.. include:: ../common.rst

API
===

Useful helpers
--------------

.. doxygenstruct:: llama::Array
   :project: LLAMA
   :members:

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

.. doxygentypedef:: llama::GetUID
   :project: LLAMA

.. doxygenstruct:: llama::LinearBytePos
   :project: LLAMA

.. doxygenstruct:: llama::SizeOf
   :project: LLAMA

.. doxygentypedef:: llama::GetDatumElementType
   :project: LLAMA

.. doxygentypedef:: llama::GetDatumElementUID
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

Allocator
^^^^^^^^^

Mapping
^^^^^^^

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


Parallelization helpers
-----------------------

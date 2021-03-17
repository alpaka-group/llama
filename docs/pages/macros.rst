.. include:: common.rst

.. _label-view:

Macros
======

Unfortunately C++ lacks some language features to express data and function
locality as well as independence of data.

The first shortcoming is what language extensions like cuda, OpenMP, OpenACC,
you name it try to solve. The second is mostly tackled by vendor specific
compiler extension. Both define new keywords and annotations to fill those gaps.
As LLAMA tries to stay independent from specific compiler vendors and
extensions, C preprocessor macros are used to define some directives only for a
sub set of compilers but with a unified interface for the user. Some macros can
even be overwritten from the outside to enable interoperability with libraries
such as alpaka.

Function locality
-----------------

Every method which can be used on offloading devices (e.g. GPUs) uses the :cpp:`LLAMA_FN_HOST_ACC_INLINE` macro as attribute.
By default it is defined as:

.. code-block:: C++

    #ifndef LLAMA_FN_HOST_ACC_INLINE
        #define LLAMA_FN_HOST_ACC_INLINE inline
    #endif

When working with cuda it should be globally defined as something like :cpp:`__host__ __device__ inline`.
Please specify this as part of your CXX flags globally.
When LLAMA is used in conjunction with alpaka, please define it as :cpp:`ALPAKA_FN_ACC __forceinline__` (with CUDA) or :cpp:`ALPAKA_FN_ACC inline`.

Data (in)dependence
-------------------

Compilers usually cannot assume that two data regions are
independent if the data is not fully visible to the compiler (e.g. a value completely lying on the stack).
One solution in C was the :cpp:`restrict` keyword which specifies that the memory pointed to by a pointer is not aliased by anything else.
However this does not work for more complex data structures containing pointers, and easily fails in other scenarios as well.
The :cpp:`restrict` keyword was therefore not added to the C++ language.

Another solution are compiler specific :cpp:`#pragma`\ s which tell the compiler that
**each** data access inside a loop can be assumed to be independent of each other.
This is handy and works with more complex data types, too.
So LLAMA provides a macro called :cpp:`LLAMA_INDEPENDENT_DATA` which can be put
in front of loops to tell the compiler that the data accesses in the
loop body are independent of each other -- and can savely be vectorized (which is the goal).

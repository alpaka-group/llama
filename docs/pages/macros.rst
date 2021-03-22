.. include:: common.rst

.. _label-view:

Macros
======

As LLAMA tries to stay independent from specific compiler vendors and
extensions, C preprocessor macros are used to define some directives for a
subset of compilers but with a unified interface for the user. Some macros can
even be overwritten from the outside to enable interoperability with libraries
such as alpaka.

Offloading
----------

We frequently have to deal with dialects of C++ which allow/require do specify to which target a function is compiled.
To support his use, every method which can be used on offloading devices (e.g. GPUs) uses the :cpp:`LLAMA_FN_HOST_ACC_INLINE` macro as attribute.
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

Compilers usually cannot assume that two data regions are independent of each other if the data is not fully visible to the compiler
(e.g. a value completely lying on the stack or the compiler observing the allocation call).
One solution in C is the :cpp:`restrict` keyword which specifies that the memory pointed to by a pointer is not aliased by anything else.
However this does not work for more complex data structures containing pointers, and easily fails in other scenarios as well.

Another solution are compiler specific :cpp:`#pragma`\ s which tell the compiler that
**each** memory access through a pointer inside a loop can be assumed to not interfere with other accesses through other pointers.
The usual goal is to allow vectorization.
Such :cpp:`#pragma`\ s are handy and work with more complex data types, too.
LLAMA provides a macro called :cpp:`LLAMA_INDEPENDENT_DATA` which can be put in front of loops to communicate the independence of memory accesses to the compiler.

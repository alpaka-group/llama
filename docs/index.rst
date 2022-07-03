.. image:: images/logo.svg

Low-Level Abstraction of Memory Access
======================================

LLAMA is a cross-platform C++17 template header-only library for the abstraction of memory
access patterns. It distinguishes between the view of the algorithm on
the memory and the real layout in the background. This enables performance
portability for multicore, manycore and gpu applications with the very same code.

In contrast to many other solutions LLAMA can define nested data structures of
arbitrary depths. It is not limited to struct of array and array of struct
data layouts but also capable to explicitly define memory layouts with padding, blocking,
striding or any other run time or compile time access pattern.

To achieve this goal LLAMA is split into mostly independent, orthogonal parts
completely written in modern C++17 to run on as many architectures and with as
many compilers as possible while still supporting extensions needed e.g. to run
on GPU or other many core hardware.

LLAMA is licensed under the LGPL3+.

.. toctree::
   :caption: USER DOCUMENTATION
   :maxdepth: 2

   pages/install
   pages/introduction
   pages/dimensions
   pages/views
   pages/recordref
   pages/iteration
   pages/mappings
   pages/proxyreferences
   pages/blobs
   pages/copying
   pages/macros
   pages/api
   pages/llama_vs_cpp

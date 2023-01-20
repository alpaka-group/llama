.. image:: images/logo.svg

Low-Level Abstraction of Memory Access
======================================

LLAMA is a cross-platform C\++17/C\++20 header-only template library for the abstraction of data layout and memory access.
It separtes the view of the algorithm on the memory and the real data layout in the background.
This allows for performance portability in applications running on heterogeneous hardware with the very same code.

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
   pages/simd
   pages/macros
   pages/api
   pages/llama_vs_cpp

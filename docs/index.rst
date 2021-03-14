.. LLAMA documentation master file, created by
   sphinx-quickstart on Wed Sep 26 13:28:02 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: images/logo.svg

Low Level Abstraction of Memory Access
======================================

LLAMA is a C++17 template header-only library for the abstraction of memory
access patterns. It distinguishes between the view of the algorithm on
the memory and the real layout in the background. This enables performance
portability for multicore, manycore and gpu applications with the very same code.

In contrast to many other solutions LLAMA can define nested data structures of
arbitrary depths and is not limited only to struct of array and array of struct
data layouts but is also capable to explicitly define padding, blocking,
striding and any other run time or compile time access pattern simultaneously.

To archieve this goal LLAMA is split into mostly independent, orthogonal parts
completely written in modern C++17 to run on as many architectures and with as
many compilers as possible while still supporting extensions needed e.g. to run
on GPU or other many core hardware.

LLAMA is licensed under the LGPL3+.

.. toctree::
   :caption: USER DOCUMENTATION
   :maxdepth: 2

   pages/install
   pages/introduction
   pages/domains
   pages/views
   pages/mappings
   pages/blobs
   pages/api

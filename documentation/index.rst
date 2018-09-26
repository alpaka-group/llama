.. LLAMA documentation master file, created by
   sphinx-quickstart on Wed Sep 26 13:28:02 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. only:: html

  .. image:: images/logo.svg

.. only:: latex

  .. image:: images/logo.pdf

Low Level Abstraction of Memory Access
======================================

LLAMA is a C++11 template header-only library for the abstraction of memory
access patterns. It distinguishs between the view of the algorithm on
the memory and the real layout in the background. This enables performance
portability for multicore, manycore and gpu application with the very same code.
LLAMA is licensed under the LGPL2+.

.. toctree::
   :caption: INSTALLATION
   :maxdepth: 2

   pages/install

.. toctree::
   :caption: USER DOCUMENTATION
   :maxdepth: 2

   pages/motivation
   pages/concept
   pages/domains
   pages/views
   pages/mappings
   pages/allocators
   pages/plans

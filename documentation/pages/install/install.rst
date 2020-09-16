.. include:: ../common.rst

Installation
============

Getting LLAMA
-------------

The most recent version of LLAMA can be found at
`GitHub <https://github.com/ComputationalRadiationPhysics/llama>`_.

All examples use CMake and the library itself provides a
:bash:`llama-config.cmake` to be found by CMake. Although LLAMA is a header-only
library, it provides installation capabilities for this file, the header files,
but also for the optionally built examples.

Requirements
------------

LLAMA itself only needs boost version 1.66.0 or higher. Furthermore some
examples need the recent develop version of
`alpaka <https://github.com/ComputationalRadiationPhysics/alpaka>`_ for
demonstrating the inter library collaboration.

Building the examples
---------------------

As LLAMA is using CMake the examples can be easily built with

.. code-block:: bash

	mkdir build
	cd build
	cmake ..

This will search for all requirements. If some requirements for examples are not
found, the building (and installation) of these examples is deactivated. It can
be activated again with

.. code-block:: bash

	ccmake ..

This can also be used to add search paths after the initial call to cmake for missing libraries and
to deactivate the build of all examples, especially if they should not be
installed. Otherwise :bash:`make install` will install the llama-config.cmake,
the include directory of LLAMA **and** all built examples.

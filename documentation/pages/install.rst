.. role:: bash(code)
   :language: bash

Getting LLAMA
=============

The most recent version of LLAMA can be found at
`GitHub <https://github.com/ComputationalRadiationPhysics/llama>`_.

All examples use CMake and the library itself provides a llama-config.cmake to
be found by CMake. Although LLAMA is a header-only library, it provies
installation capabilities for this file, of course the includes, but also for
built examples.

Requirements
============

LLAMA itself only needs boost version 1.66.0 or higher as the quite new
boost::mp11 template programming library is needed. Furthermore some examples
need the recent develop version of
`alpaka <https://github.com/ComputationalRadiationPhysics/alpaka>`_ for
demonstrating the inter library collaboration and/or
`png++ <https://www.nongnu.org/pngpp/>`_ for saving some generated images.

Building the examples
=====================

As LLAMA is using CMake the examples can be easily build with

.. code-block:: bash

	mkdir build
	cd build
	cmake ..

This will search for all requirements. If some requirements for examples are not
found, the building (and installation) of these examples is deactivated. It can
be activated again with

.. code-block:: bash

	ccmake ..

This can also be used to afterwards add search paths for missing libraries and
to deactivate the building of all examples if they shall not be installed.
Otherwise :bash:`make install` will install the llama-config.cmake, the include
directory of LLAMA **and** all built examples.

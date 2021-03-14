.. include:: common.rst

Installation
============

Getting LLAMA
-------------

The most recent version of LLAMA can be found at
`GitHub <https://github.com/alpaka-group/llama>`_.

All examples use CMake and the library itself provides a
:bash:`llama-config.cmake` to be found by CMake. Although LLAMA is a header-only
library, it provides installation capabilities for this file, the header files,
but also for the optionally built examples.

Dependencies
------------

 - Boost 1.66.0 or higher
 - libfmt 6.2.1 or higher
 - `Alpaka <https://github.com/alpaka-group/alpaka>`_ (optional) for building some examples
 - `Vc <https://github.com/VcDevel/Vc>`_ (optional) for building some examples


Building the examples
---------------------

As LLAMA is using CMake the examples can be easily built with

.. code-block:: bash

	mkdir build
	cd build
	cmake ..

This will search for all depenencies and create a build system for your platform.
If Alpaka or Vc is not found, all Alpaka or Vc examples will be disabled for building and installing.


CMake settings after the initial generation may be changed again with:

.. code-block:: bash

	ccmake ..

This can also be used to add search paths after the initial call to cmake for missing libraries and to deactivate the build and installation of examples.
Finally, run :bash:`make install` or your platforms equivalent to install the llama-config.cmake, the header files of LLAMA **and** all built examples.

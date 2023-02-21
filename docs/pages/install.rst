.. include:: common.rst

Installation
============

Getting LLAMA
-------------

The most recent version of LLAMA can be found at `GitHub <https://github.com/alpaka-group/llama>`_.

.. code-block:: bash

	git clone https://github.com/alpaka-group/llama
	cd llama

All examples use CMake and the library itself provides a :bash:`llama-config.cmake` to be found by CMake.
Although LLAMA is a header-only library, it provides installation capabilities via CMake.

Dependencies
------------

LLAMA library
^^^^^^^^^^^^^

At its core, using the LLAMA library requires:

- cmake 3.18.3 or higher
- Boost 1.74.0 or higher
- libfmt 6.2.1 or higher (optional) for support to dump mappings as SVG/HTML

Tests
^^^^^

Building the unit tests additionally requires:

- Catch2 3.0.1 or higher

Examples
^^^^^^^^

To build all examples of LLAMA, the following additional libraries are needed:

- libfmt 6.2.1 or higher
- `Alpaka <https://github.com/alpaka-group/alpaka>`_ 0.9.0 or higher
- `xsimd <https://github.com/xtensor-stack/xsimd>`_ 9.0.1 or higher
- `ROOT <https://root.cern/>`_
- `tinyobjloader <https://github.com/tinyobjloader/tinyobjloader>`_ 2.0.0-rc9 or higher


Build tests and examples
------------------------

As LLAMA is using CMake the tests and examples can be easily built with:

.. code-block:: bash

	mkdir build
	cd build
	cmake .. -DBUILD_TESTING=ON -DLLAMA_BUILD_EXAMPLES=ON
	ccmake .. // optionally change configuration after first run of cmake
	cmake --build .

This will search for all dependencies and create a build system for your platform.
If necessary dependencies are not found, the corresponding examples will be disabled.
After the initial call to `cmake`, `ccmake` can be used to add search paths for missing libraries and to deactivate building tests and examples.

Install LLAMA
-------------

To install LLAMA on your system, you can run (with privileges):

.. code-block:: bash

	cmake --install .

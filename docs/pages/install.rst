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

 - Boost 1.66.0 or higher
 - libfmt 6.2.1 or higher (optional) for building some examples and LLAMA supporting to dump mappings as SVG/HTML
 - `Alpaka <https://github.com/alpaka-group/alpaka>`_ (optional) for building some examples
 - `Vc <https://github.com/VcDevel/Vc>`_ (optional) for building some examples


Build tests and examples
------------------------

As LLAMA is using CMake the tests and examples can be easily built with:

.. code-block:: bash

	mkdir build
	cd build
	cmake ..
	ccmake .. // optionally change configuration after first run of cmake
	cmake --build .

This will search for all depenencies and create a build system for your platform.
If Alpaka or Vc is not found, the corresponding examples will be disabled.
After the initial call to `cmake`, `ccmake` can be used to add search paths for missing libraries and to deactivate building tests and examples.
The tests can be disabled by setting `BUILD_TESTING` to `OFF` (default: `ON`).
The examples can be disabled by setting `LLAMA_BUILD_EXAMPLES` to `OFF` (default: `ON`).

Install LLAMA
-------------

To install LLAMA on your system, you can run (with privileges):

.. code-block:: bash

	sudo cmake --install .

LLAMA – Low Level Abstraction of Memory Access
==============================================

![LLAMA](documentation/images/logo_400x169.png)

LLAMA is a C++17 template header-only library for the abstraction of memory
access patterns. It distinguishes between the view of the algorithm on
the memory and the real layout in the background. This enables performance
portability for multicore, manycore and gpu applications with the very same code.

In contrast to many other solutions LLAMA can define nested data structures of
arbitrary depths and is not limited only to struct of array and array of struct
data layouts but is also capable to explicitly define padding, blocking,
striding and any other run time or compile time access pattern simultaneously.

To archieve this goal LLAMA is splitted in mostly independent, orthogonal parts
completely written in modern C++17 to run on as many architectures and with as
many compilers as possible while still supporting extensions needed e.g. to run
on GPU or other many core hardware.

The user documentation and an overview about the concepts and ideas can be found
here:
https://llama-doc.rtfd.io

Doxygen generated API documentation is located here:
https://alpaka-group.github.io/llama/

Rules for contributions can be found in [CONTRIBUTING.md](CONTRIBUTING.md).

LLAMA is licensed under the LGPL3+.

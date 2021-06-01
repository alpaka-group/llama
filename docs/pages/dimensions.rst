.. include:: common.rst

.. _label-dimensions:

Dimensions
==========

As mentioned in the section before, LLAMA distinguishes between the array and the record dimensions.
The most important difference is that the array dimensions are defined at *run time* whereas the record dimension is defined at *compile time*.
This allows to make the problem size itself a run time value but leaves the compiler room to optimize the data access.

.. _label-ad:

Array dimensions
----------------

The array dimensions are an :math:`N`-dimensional array with :math:`N` itself being a
compile time value but with run time values inside. LLAMA brings its own
:ref:`array class <label-api-array>` for such kind of data structs which is
ready for interoperability with hardware accelerator C++ dialects such as CUDA
(Nvidia) or HIP (AMD), or abstraction libraries such as the already mentioned
alpaka.

A definition of three array dimensions of the size :math:`128 \times 256 \times 32` looks like this:

.. code-block:: C++

    llama::ArrayDims arrayDimsSize{128, 256, 32};

The template arguments are deduced by the compiler using `CTAD <https://en.cppreference.com/w/cpp/language/class_template_argument_deduction>`_.
The full type of :cpp:`arrayDimsSize` is :cpp:`llama::ArrayDims<3>`.

.. _label-rd:

Record dimension
----------------

The record dimension is a tree structure completely defined at compile time.
Nested C++ structs, which the record dimension tries to abstract, they are trees too.
Let's have a look at this simple example struct for storing a pixel value:

.. code-block:: C++

    struct Pixel {
        struct {
            float r
            float g
            float b;
        } color;
        char alpha;
    };

This defines this tree

.. image:: ../images/layout_tree.svg

Unfortunately with C++ it is not possible yet to "iterate" over a struct at compile time and extract member types and names,
as it would be needed for LLAMA's mapping (although there are proposals to provide such a facility).
For now LLAMA needs to define such a tree itself using two classes, :cpp:`llama::Record` and :cpp:`llama::Field`.
:cpp:`llama::Record` is a compile time list of :cpp:`llama::Field`.
:cpp:`llama::Field` has a name and a fundamental type **or** another :cpp:`llama::Record` list of child :cpp:`llama::Field`\ s.
The name of a :cpp:`llama::Field` needs to be C++ type as well.
We recommend creating empty tag types for this.
These tags serve as names when describing accesses later.
Furthermore, these tags also enable a semantic binding even between two different record dimensions.

To make the code easier to read, the following shortcuts are defined:

* :cpp:`llama::Record` → :cpp:`llama::Record`
* :cpp:`llama::Field` → :cpp:`llama::Field`

A record dimension itself is just a :cpp:`llama::Record` (or a fundamental type), as seen here for the given tree:

.. code-block:: C++

    struct color {};
    struct alpha {};
    struct r {};
    struct g {};
    struct b {};

    using RGB = llama::Record<
        llama::Field<r, float>,
        llama::Field<g, float>,
        llama::Field<b, float>
    >;
    using Pixel = llama::Record<
        llama::Field<color, RGB>,
        llama::Field<alpha, char>
    >;

Arrays of compile-time extent are also supported as arguments to :cpp:`llama::Field`.
Such arrays are expanded into a :cpp:`llama::Record` with multiple :cpp:`llama::Field`\ s of the same type.
E.g. :cpp:`llama::Field<Tag, float[4]>` is expanded into

.. code-block:: C++

    llama::Field<Tag, llama::Record<
        llama::Field<llama::RecordCoord<0>, float>,
        llama::Field<llama::RecordCoord<1>, float>,
        llama::Field<llama::RecordCoord<2>, float>,
        llama::Field<llama::RecordCoord<3>, float>
    >>

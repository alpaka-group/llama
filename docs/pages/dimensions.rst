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

The array dimensions form an :math:`N`-dimensional array with :math:`N` itself being a compile time value.
The extent of each dimension can be a compile time or runtime values.

A simple definition of three array dimensions of the extents :math:`128 \times 256 \times 32` looks like this:

.. code-block:: C++

    llama::ArrayExtents extents{128, 256, 32};

The template arguments are deduced by the compiler using `CTAD <https://en.cppreference.com/w/cpp/language/class_template_argument_deduction>`_.
The full type of :cpp:`extents` is :cpp:`llama::ArrayExtents<llama::dyn, llama::dyn, llama::dyn>`.

By explicitely specifying the template arguments, we can mix compile time and runtime extents, where the constant :cpp:`llama::dyn` denotes a dynamic extent:

.. code-block:: C++

    llama::ArrayExtents<llama::dyn, 256, llama::dyn> extents{128, 32};

The template argument list specifies the order and nature (compile vs. runtime) of the extents.
An instance of :cpp:`llama::ArrayExtents` can then be constructed with as many runtime extents as :cpp:`llama::dyn`\ 's specified in the template argument list.

By setting a specific value for all template arguments, the array extents are fully determined at compile time.

.. code-block:: C++

    llama::ArrayExtents<128, 256, 32> extents{};

This is important if such extents are later embedded into other LLAMA objects such as mappings or views, where they should not occupy any additional memory.

.. code-block:: C++

    llama::ArrayExtents<128, 256, 32> extents{};
    static_assert(sizeof(extents) == 1); // no object can have size 0
    struct S : llama::ArrayExtents<128, 256, 32> { char c; } s;
    static_assert(sizeof(s) == sizeof(char)); // empty base optimization eliminates storage

To later described indices into the array dimensions described by a :cpp:`llama::ArrayExtents`, an instance of :cpp:`llama::ArrayIndex` is used:

.. code-block:: C++

    llama::ArrayIndex i{2, 3, 4};
    // full type of i: llama::ArrayIndex<3>

Contrary to :cpp:`llama::ArrayExtents` which can store a mix of compile and runtime values, :cpp:`llama::ArrayIndex` only stores runtime indices, so it is templated on the number of dimensions.
This might change at some point in the future, if we find sufficient evidence that a design similar to :cpp:`llama::ArrayExtents` is also useful for :cpp:`llama::ArrayIndex`.

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

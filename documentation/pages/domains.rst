.. include:: common.rst

.. _label-domains:

Domains
=======

As mentioned in the section before, LLAMA distinguishes between the array and the
datum domain. The most important difference is that the array domain is defined
at *run time* whereas the datum domain is defined at *compile time*. This allows
to make the problem size itself a run time value but leaves the compiler room
to optimize the data access.

.. _label-ad:

Array domain
-----------

The array domain is an :math:`N`-dimensional array with :math:`N` itself being a
compile time value but with run time values inside. LLAMA brings its own
:ref:`array class <label-api-array>` for such kind of data structs which is
ready for interoperability with hardware accelerator C++ dialects such as CUDA
(Nvidia) or HIP (AMD), or abstraction libraries such as the already mentioned
alpaka.

A definition of a three-dimensional array domain of the size
:math:`128 \times 256 \times 32` looks like this:

.. code-block:: C++

    llama::ArrayDomain arrayDomainSize{128, 256, 32};

The template arguments are deduced by the compiler using `CTAD <https://en.cppreference.com/w/cpp/language/class_template_argument_deduction>`_.
The full type of :cpp:`arrayDomainSize` is :cpp:`llama::ArrayDomain<3>`.

.. _label-dd:

Datum domain
------------

The datum domain is a tree structure completely defined at compile time.
Nested C++ structs, which the datum domain tries to abstract, they are trees too.
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
For now LLAMA needs to define such a tree itself using two classes, :cpp:`DatumStruct` and :cpp:`DatumElement`.
:cpp:`DatumStruct` is a compile time list of :cpp:`DatumElement`.
:cpp:`DatumElement` has a name and a fundamental type **or** another :cpp:`DatumStruct` list of child :cpp:`DatumElement`\ s.
The name of a :cpp:`DatumElement` needs to be C++ type as well.
We recommend creating empty tag types for this.
These tags serve as names when describing accesses later.
Furthermore, these tags also enable a semantic binding even between two different datum domains.

To make the code easier to read, the following shortcuts are defined:

* :cpp:`llama::DatumStruct` → :cpp:`llama::DS`
* :cpp:`llama::DatumElement` → :cpp:`llama::DE`

A datum domain itself is just a :cpp:`DatumStruct` (or a fundamental type), as seen here for the given tree:

.. code-block:: C++

    struct color {};
    struct alpha {};
    struct r {};
    struct g {};
    struct b {};

    using Pixel = llama::DS<
        llama::DE<color, llama::DS<
            llama::DE<r, float>,
            llama::DE<g, float>,
            llama::DE<b, float>
        >>,
        llama::DE<alpha, char>
    >;

A :cpp:`DatumArray` is essentially a :cpp:`DatumStruct` with multiple :cpp:`DatumElement`\ s of the same type.
E.g. :cpp:`DatumArray<float, 4>` is the same as

.. code-block:: C++

    llama::DS<
        llama::DE<llama::Index<0>, float>,
        llama::DE<llama::Index<1>, float>,
        llama::DE<llama::Index<2>, float>,
        llama::DE<llama::Index<3>, float>
    >

LLAMA also defines a shortcuts for a datum array:

* :cpp:`llama::DatumArray` → :cpp:`llama::DA`

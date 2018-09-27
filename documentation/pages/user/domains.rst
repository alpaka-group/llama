.. include:: ../common.rst

.. _label-domains:

Domains
=======

As mentioned in the section before, LLAMA differs between the user and the
datum domain. The most important difference is that the user domain is defined
as *run time* whereas the datum domain is defined at compile time. This allows
to make the problem size itself a run time value but leaves the compiler room
to optimize the data access.

.. _label-ud:

User domain
-----------

The user domain is an :math:`N`-dimensional array with :math:`N` itself being a
compile time value but with run time values inside. LLAMA brings an own array
class for such kind of data structs which are ready for interoperability with
hardware accelerator C++ dialects as CUDA (Nvidia) or HIP (AMD), or abstracting
libraries as the already mentioned alpaka library.

A definition of a three-dimensional user domain of the size
:math:`128 \times 256 \times 32` looks like this:

.. code-block:: C++

    using UserDomain = llama::UserDomain< 3 >;
    const UserDomain userDomainSize{ 128, 256, 32 };

.. _label-dd:

Datum domain
------------

The completely compile time defined datum domain is basically a tree structure.
If we have a look at C++ structs (which the datum domain tries to abstract)
they basically are trees, too. Let's have a look at this simple example struct
for storing a pixel value

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

.. only:: html

  .. image:: ../../images/layout_tree.svg

.. only:: latex

  .. image:: ../../images/layout_tree.pdf

Unfortunately with C++ it is not possible yet to "iterate" over a struct at
compile time as it is needed for LLAMA's mapping. Even if recent versions
of the language will start to support such a feature because of the many
compiler vendors LLAMA tries to support, the library is stuck to the C++11
feature set. So LLAMA needs to define such a tree itself. For this two
classes are defined, :cpp:`DatumStruct` and :cpp:`DatumElement`.
:cpp:`DatumStruct` is a compile time list of :cpp:`DatumElement`, which has a
name and a type **or** another :cpp:`DatumStruct` list of child
:cpp:`DatumElement`\ s. The names of the :cpp:`DatumElement`\ s need to be
predefined as structs. This enables a semantic binding even between two
different datum domains.

A datum domain itself is just such a :cpp:`DatumStruct` as seen here for the
given tree:

.. code-block:: C++

    struct color {};
    struct alpha {};
    struct r {};
    struct g {};
    struct b {};

    using Pixel = llama::DatumStruct <
        llama::DatumElement < color, llama::DatumStruct <
            llama::DatumElement < r, float >,
            llama::DatumElement < g, float >,
            llama::DatumElement < b, float >
        > >,
        llama::DatumElement < alpha, char >
    >;

Furthermore a third class :cpp:`DatumArray` is defined, which can be used to
define a :cpp:`DatumStruct` with multiple, nameless :cpp:`DatumElement`\ s of
the same type, e.g. :cpp:`llama::DatumArray < float, 4 >` is the same as

.. code-block:: C++

    llama::DatumStruct <
        llama::DatumElement < llama::NoName, float >,
        llama::DatumElement < llama::NoName, float >,
        llama::DatumElement < llama::NoName, float >,
        llama::DatumElement < llama::NoName, float >
    >

To make the code easier to read, shortcuts are defined for each of these
classes:

* :cpp:`llama::DatumStruct` → :cpp:`llama::DS`
* :cpp:`llama::DatumElement` → :cpp:`llama::DE`
* :cpp:`llama::DatumArray` → :cpp:`llama::DA`

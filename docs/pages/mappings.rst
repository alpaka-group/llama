.. include:: common.rst

.. _label-mappings:

Mappings
========

One of the core tasks of LLAMA is to map an address from the array and record dimensions to some address in the allocated memory space.
This is particularly challenging if the compiler shall still be able to optimize the resulting memory accesses (vectorization, reordering, aligned loads, etc.).
The compiler needs to **understand** the semantic of the mapping at compile time.
Otherwise the abstraction LLAMA provides will perform poorly.
Thus, mappings are compile time parameters to LLAMA's views (and e.g. not hidden behind a virtual dispatch).
LLAMA provides several ready-to-use mappings, but users are also free to supply their own mappings.

.. image:: ../images/mapping.svg

LLAMA supports and uses different classes of mapping that differ in their usage:

Physical mappings
^^^^^^^^^^^^^^^^^

A physical mapping is the primary form of a mapping.
Mapping a record coordinate and array dimension index through a physical mapping results in a blob number and offset.
This information is then used either by a view or subsequent mapping and, given a blob array, can be turned into a physical memory location,
which is provided as l-value reference to the mapped field type of the record dimension.

Computed mappings
^^^^^^^^^^^^^^^^^

A computed mapping may invoke a computation to map a subset of the record dimension.
The fields of the record dimension which are mapped using a computation, are called computed fields.
A computed mapping does not return a blob number and offset for computed fields, but rather a reference to memory directly.
However, this reference is not an l-value reference but a :ref:`proxy reference <label-proxyreferences>`,
since this reference needs to encapsulate computations to be performed when reading or writing through the reference.
For non-computed fields, a computed mapping behaves like a physical mapping.
A mapping with only computed fields is called a fully computed mapping, otherwise a partially computed mapping.

Meta mappings
^^^^^^^^^^^^^

A meta mapping is a mapping that builds on other mappings.
Examples are altering record or array dimensions before passing the information to another mapping
or modifying the blob number and offset returned from a mapping.
A meta mapping can also instrument or trace information on the accesses to another mapping.
Meta mappings are orthogonal to physical and computed mappings.


Concept
-------

A LLAMA mapping is used to create views as detailed in the :ref:`allocView API section <label-api-allocView>`
and views consult the mapping when resolving accesses.
The view requires each mapping to fulfill at least the following concept:

.. code-block:: C++

    template <typename M>
    concept Mapping = requires(M m) {
        typename M::ArrayExtents;
        typename M::RecordDim;
        { m.extents() } -> std::same_as<typename M::ArrayExtents>;
        { +M::blobCount } -> std::same_as<std::size_t>;
        requires isConstexpr<M::blobCount>;
        { m.blobSize(std::size_t{}) } -> std::same_as<typename M::ArrayExtents::value_type>;
    };

That is, each mapping type needs to expose the types :cpp:`ArrayExtents` and :cpp:`RecordDim`.
Each mapping also needs to provide a getter `extents()` to retrieve the runtime value of the :cpp:`ArrayExtents` held by the mapping,
and provide a :cpp:`static constexpr` member variable :cpp:`blobCount`.
Finally, the member function :cpp:`blobSize(i)` gives the size in bytes of the :cpp:`i`\ th block of memory needed for this mapping
using the value type of the array extents.
:cpp:`i` is in the range of :cpp:`0` to :cpp:`blobCount - 1`.

Additionally, a mapping needs to be either a physical or a computed mapping.
Physical mappings, in addition to being mappings, need to fulfill the following concept:

.. code-block:: C++

    template <typename M>
    concept PhysicalMapping = Mapping<M> && requires(M m, typename M::ArrayIndex ai, RecordCoord<> rc) {
        { m.blobNrAndOffset(ai, rc) } -> std::same_as<NrAndOffset<typename M::ArrayExtents::value_type>>;
    };

That is, they must provide a member function callable as :cpp:`blobNrAndOffset(ai, rc)` that implements the core mapping logic,
which is translating an array index :cpp:`ai` and record coordinate :cpp:`rc` into a value of :cpp:`llama::NrAndOffset`,
containing the blob number of offset within the blob where the value should be stored.
The integral type used for computing blob number and offset should be the value type of the array extents.

..
    Computed mappings, in addition to being mappings, need to fulfill the following concept:

    .. code-block:: C++

        template <typename M>
        concept ComputedMapping = Mapping<M> && requires(M m, typename M::ArrayIndex ai, RecordCoord<> rc) {
            { m.isComputed(rc) } -> std::same_as<bool>;
            { m.compute(ai, rc, Array<Array<std::byte, 0>, 0>{}) } -> AnyReferenceTo<GetType<typename M::RecordDim, RC>>;
            { m.blobNrAndOffset(ai, rc) } -> std::same_as<NrAndOffset<typename M::ArrayExtents::value_type>>;
        };

    That is, they must provide a :cpp:`constexpr` member function :cpp:`isComputed(rc)`, callable for a record coordinate,
    that returns a bool indicating whether or not the record field indicated by the given record coordinate is computed or not.


AoS
---

LLAMA provides a family of AoS (array of structs) mappings based on a generic implementation.
AoS mappings keep the data of a single record close together and therefore maximize locality for accesses to an individual record.
However, they do not vectorize well in practice.

.. code-block:: C++

    llama::mapping::AoS<ArrayExtents, RecordDim> mapping{extents}; 
    llama::mapping::AoS<ArrayExtents, RecordDim, false> mapping{extents}; // pack fields (violates alignment)
    llama::mapping::AoS<ArrayExtents, RecordDim, false
        llama::mapping::LinearizeArrayIndexLeft> mapping{extents}; // pack fields, column major

By default, the array dimensions spanned by :cpp:`ArrayExtents` are linearized using :cpp:`llama::mapping::LinearizeArrayIndexRight`.
LLAMA provides the aliases :cpp:`llama::mapping::AlignedAoS` and :cpp:`llama::mapping::PackedAoS` for convenience.


SoA
---

LLAMA provides a family of SoA (struct of arrays) mappings based on a generic implementation.
SoA mappings store the attributes of a record contiguously and therefore maximize locality for accesses to the same attribute of multiple records.
This layout auto-vectorizes well in practice.

.. code-block:: C++

    llama::mapping::SoA<ArrayExtents, RecordDim> mapping{extents};
    llama::mapping::SoA<ArrayExtents, RecordDim, true> mapping{extents}; // separate blob for each attribute
    llama::mapping::SoA<ArrayExtents, RecordDim, true,
        llama::mapping::LinearizeArrayIndexLeft> mapping{extents}; // separate blob for each attribute, column major

By default, the array dimensions spanned by :cpp:`ArrayExtents` are linearized using :cpp:`llama::mapping::LinearizeArrayIndexRight` and the layout is mapped into a single blob.
LLAMA provides the aliases :cpp:`llama::mapping::SingleBlobSoA` and :cpp:`llama::mapping::MultiBlobSoA` for convenience.


AoSoA
-----

There are also combined AoSoA (array of struct of arrays) mappings.
Since the mapping code is more complicated, compilers currently fail to auto vectorize view access.
We are working on this.
The AoSoA mapping has a mandatory additional parameter specifying the number of elements which are blocked in the inner array of AoSoA.

.. code-block:: C++

    llama::mapping::AoSoA<ArrayExtents, RecordDim, 8> mapping{extents}; // inner array has 8 values
    llama::mapping::AoSoA<ArrayExtents, RecordDim, 8,
        llama::mapping::LinearizeArrayIndexLeft> mapping{extents}; // inner array has 8 values, column major

By default, the array dimensions spanned by :cpp:`ArrayExtents` are linearized using :cpp:`llama::mapping::LinearizeArrayIndexRight`.

LLAMA also provides a helper :cpp:`llama::mapping::maxLanes` which can be used to determine the maximum vector lanes which can be used for a given record dimension and vector register size.
In this example, the inner array a size of N so even the largest type in the record dimension can fit N times into a vector register of 256bits size (e.g. AVX2).

.. code-block:: C++

    llama::mapping::AoSoA<ArrayExtents, RecordDim,
        llama::mapping::maxLanes<RecordDim, 256>> mapping{extents};


One
---

The One mapping is intended to map all coordinates in the array dimensions onto the same memory location.
This is commonly used in :cpp:`llama::One`, but also offers interesting applications in conjunction with the :cpp:`llama::mapping::Split` mapping.


Split
-----

The Split mapping is a meta mapping.
It transforms the record dimension and delegates mapping to other mappings.
Using a record coordinate, a tag list, or a list of record coordinates or a list of tag lists,
a subtree of the record dimension is selected and mapped using one mapping.
The remaining record dimension is mapped using a second mapping.

.. code-block:: C++

    llama::mapping::Split<ArrayExtents, RecordDim,
        llama::RecordCoord<1>, llama::mapping::SoA, llama::mapping::PackedAoS>
            mapping{extents}; // maps the subtree at index 1 as SoA, the rest as packed AoS

Split mappings can be nested to map a record dimension into even fancier combinations.


Heatmap
-------

The Heatmap mapping is a meta mapping that wraps over an inner mapping and counts all accesses made to all bytes.
A script for gnuplot visualizing the heatmap can be extracted.

.. code-block:: C++

    auto anyMapping = ...;
    llama::mapping::Heatmap mapping{anyMapping};
    ...
    mapping.writeGnuplotDataFileBinary(view.blobs(), std::ofstream{"heatmap.data", std::ios::binary});
    std::ofstream{"plot.sh"} << mapping.gnuplotScriptBinary;


FieldAccessCount
----------------

The FieldAccessCount mapping is a meta mapping that wraps over an inner mapping and counts all accesses made to the fields of the record dimension.
A report is printed to stdout when requested.
The mapping adds an additional blob to the blobs of the inner mapping used as storage for the access counts.

.. code-block:: C++

    auto anyMapping = ...;
    llama::mapping::FieldAccessCount mapping{anyMapping};
    ...
    mapping.printFieldHits(view.blobs()); // print report with read and writes to each field

The FieldAccessCount mapping uses proxy references to instrument reads and writes.
If this is problematic, it can also be configured to return raw C++ references.
In that case, only the number of memory location computations can be traced,
but not how often the program reads/writes to those locations.
Also, the data type used to count accesses is configurable.

.. code-block:: C++

    auto anyMapping = ...;
    llama::mapping::FieldAccessCount<decltype(anyMapping), std::size_t, false> mapping{anyMapping};


Null
----

The Null mappings is a fully computed mapping that maps all elements to nothing.
Writing data through a reference obtained from the Null mapping discards the value.
Reading through such a reference returns a default constructed object.
A Null mapping requires no storage and thus its :cpp:`blobCount` is zero.

.. code-block:: C++

    llama::mapping::Null<ArrayExtents, RecordDim> mapping{extents};


Bytesplit
---------

The Bytesplit mapping is a computed meta mapping that wraps over an inner mapping.
It transforms the record dimension by replacing each field type by a byte array of the same size before forwarding the record dimension to the inner mapping.

.. code-block:: C++

    template <typename RecordDim, typename ArrayExtents>
    using InnerMapping = ...;

    llama::mapping::Bytesplit<ArrayExtents, RecordDim, InnerMapping>
            mapping{extents};


Byteswap
---------

The Byteswap mapping is a computed meta mapping that wraps over an inner mapping.
It swaps the bytes of all values when reading/writing.

.. code-block:: C++

    template <typename RecordDim, typename ArrayExtents>
    using InnerMapping = ...;

    llama::mapping::Byteswap<ArrayExtents, RecordDim, InnerMapping>
            mapping{extents};


ChangeType
----------

The ChangeType mapping is a computed meta mapping that allows to change data types of several fields in the record dimension before
and mapping the adapted record dimension with a further mapping.

.. code-block:: C++

    template <typename RecordDim, typename ArrayExtents>
    using InnerMapping = ...;

    using ReplacementMap = mp_list<
        mp_list<int, short>,
        mp_list<double, float>
    >;
    llama::mapping::ChangeType<ArrayExtents, RecordDim, InnerMapping, ReplacementMap>
            mapping{extents};

In this example, all fields of type :cpp:`int` in the record dimension will be stored as :cpp:`short`,
and all fields of type :cpp:`double` will be stored as :cpp:`float`.
Conversion between the data types is done on loading and storing through a proxy reference returned from the mapping.


Projection
----------

The Projection mapping is a computed meta mapping that allows to apply a function on load/store from/two selected fields in the record dimension.
These functions are allowed to change the data type of fields in the record dimension.
The modified record dimension is then mapped with a further mapping.

.. code-block:: C++

    template <typename RecordDim, typename ArrayExtents>
    using InnerMapping = ...;

    struct Sqrt {
        static auto load(float v) -> double {
            return std::sqrt(v);
        }

        static auto store(double d) -> float {
            return static_cast<float>(d * d);
        }
    };

    using ReplacementMap = mp_list<
        mp_list<double, Sqrt>,
        mp_list<RecordCoord<0, 1>, Sqrt>
    >;
    llama::mapping::ChangeType<ArrayExtents, RecordDim, InnerMapping, ReplacementMap>
            mapping{extents};

In this example, all fields of type :cpp:`double`, and the field at coordinate RecordCoord<0, 1>, in the record dimension will store the product with itself as :cpp:`float`.
The load/store functions are called on loading and storing through a proxy reference returned from the mapping.


BitPackedIntAoS/BitPackedIntSoA
-------------------------------

The BitPackedIntSoA and BitPackedIntAoS mappings are fully computed mappings that bitpack integral values to reduce size and precision.
The bits are stored as array of structs and struct of arrays, respectively.
The number of bits used per integral is configurable.
All field types in the record dimension must be integral.

.. code-block:: C++

    unsigned bits = 7;
    llama::mapping::BitPackedIntSoA<ArrayExtents, RecordDim>
            mapping{bits, extents}; // use 7 bits for each integral in RecordDim


BitPackedFloatAoS/BitPackedFloatSoA
-----------------------------------

The BitPackedFloatAoS and BitPackedFloatSoA mappings are fully computed mapping that bitpack floating-point values to reduce size and precision.
The bits are stored as array of structs and struct of arrays, respectively.
The number of bits used to store the exponent and mantissa is configurable.
All field types in the record dimension must be floating-point.
These mappings require the C++ implementation to use `IEEE 754 <https://en.wikipedia.org/wiki/IEEE_754>`_ floating-point formats.

.. code-block:: C++

    unsigned exponentBits = 4;
    unsigned mantissaBits = 7;
    llama::mapping::BitPackedFloatSoA<ArrayExtents, RecordDim>
            mapping{exponentBits, mantissaBits, extents}; // use 1+4+7 bits for each floating-point in RecordDim


PermuteArrayIndex
-----------------

The PermuteArrayIndex mapping is a meta mapping that wraps over an inner mapping.
It permutes the array indices before passing the index information to the inner mapping.

.. code-block:: C++

    using InnerMapping = ...;

    llama::mapping::PermuteArrayIndex<InnerMapping, 2, 0, 1> mapping{extents};
    auto view = llama::allocView(mapping);
    view(1, 2, 3); // will pass {3, 1, 2} to inner mapping


Dump visualizations
-------------------

Sometimes it is hard to image how data will be laid out in memory by a mapping.
LLAMA can create a graphical representation of a mapping instance as SVG image or HTML document:

.. code-block:: C++

    std::ofstream{filename + ".svg" } << llama::toSvg (mapping);
    std::ofstream{filename + ".html"} << llama::toHtml(mapping);


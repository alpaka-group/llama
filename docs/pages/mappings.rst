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

Concept
-------

A LLAMA mapping is used to create views as detailed in the :ref:`allocView API section <label-api-allocView>` and views consult the mapping when resolving accesses.
The view requires each mapping to fulfill the following concept:

.. code-block:: C++

    template <typename M>
    concept Mapping = requires(M m) {
        typename M::ArrayDims;
        typename M::RecordDim;
        { M::blobCount } -> std::convertible_to<std::size_t>;
        llama::Array<int, M::blobCount>{}; // validates constexpr-ness
        { m.blobSize(std::size_t{}) } -> std::same_as<std::size_t>;
        { m.blobNrAndOffset(typename M::ArrayDims{}) } -> std::same_as<NrAndOffset>;
    };

That is, each mapping type needs to expose the types :cpp:`M::ArrayDims` and :cpp:`M::RecordDim`.
Furthermore, each mapping needs to provide a static constexpr member variable :cpp:`blobCount` and two member functions.
:cpp:`blobSize(i)` gives the size in bytes of the :cpp:`i`\ th block of memory needed for this mapping.
:cpp:`i` is in the range of :cpp:`0` to :cpp:`blobCount - 1`.
:cpp:`blobNrAndOffset(ad)` implements the core mapping logic by translating an array coordinate :cpp:`ad` into a value of :cpp:`llama::NrAndOffset`, containing the blob number of offset within the blob where the value should be stored.

AoS mappings
------------

LLAMA provides a family of AoS (array of structs) mappings based on a generic implementation.
AoS mappings keep the data of a single record close together and therefore maximize locality for accesses to an individual record.
However, they do not vectorize well in practice.

.. code-block:: C++

    llama::mapping::AoS<ArrayDims, RecordDim> mapping{arrayDimsSize};
    llama::mapping::AoS<ArrayDims, RecordDim, true> mapping{arrayDimsSize}; // respect alignment
    llama::mapping::AoS<ArrayDims, RecordDim, true
        llama::mapping::LinearizeArrayDimsFortran> mapping{arrayDimsSize}; // respect alignment, column major

By default, the :cpp:`ArrayDims` is linearized using :cpp:`llama::mapping::LinearizeArrayDimsCpp` and the layout is tightly packed.
LLAMA provides the aliases :cpp:`llama::mapping::AlignedAoS` and :cpp:`llama::mapping::PackedAoS` for convenience.


There is also a combined array of struct of arrays mapping,
but, since the mapping code is more complicated, compilers currently fail to auto vectorize view access:

.. code-block:: C++

    llama::mapping::AoSoA<ArrayDims, RecordDim, Lanes> mapping{arrayDimsSize};

.. _label-tree-mapping:


SoA mappings
------------

LLAMA provides a family of SoA (struct of arrays) mappings based on a generic implementation.
SoA mappings store the attributes of a record contigiously and therefore maximize locality for accesses to the same attribute of multiple records.
This layout auto vectorizes well in practice.

.. code-block:: C++

    llama::mapping::SoA<ArrayDims, RecordDim> mapping{arrayDimsSize};
    llama::mapping::SoA<ArrayDims, RecordDim, true> mapping{arrayDimsSize}; // separate blob for each attribute
    llama::mapping::SoA<ArrayDims, RecordDim, true,
        llama::mapping::LinearizeArrayDimsFortran> mapping{arrayDimsSize}; // separate blob for each attribute, column major

By default, the :cpp:`ArrayDims` is linearized using :cpp:`llama::mapping::LinearizeArrayDimsCpp` and the layout is mapped into a single blob.
LLAMA provides the aliases :cpp:`llama::mapping::SingleBlobSoA` and :cpp:`llama::mapping::MultiBlobSoA` for convenience.


AoSoA mappings
--------------

There are also combined AoSoA (array of struct of arrays) mappings.
Since the mapping code is more complicated, compilers currently fail to auto vectorize view access.
We are working on this.
The AoSoA mapping has a mandatory additional parameter specifying the number of elements which are blocked in the inner array of AoSoA.

.. code-block:: C++

    llama::mapping::AoSoA<ArrayDims, RecordDim, 8> mapping{arrayDimsSize}; // inner array has 8 values
    llama::mapping::AoSoA<ArrayDims, RecordDim, 8,
        llama::mapping::LinearizeArrayDimsFortran> mapping{arrayDimsSize}; // inner array has 8 values, column major

By default, the :cpp:`ArrayDims` is linearized using :cpp:`llama::mapping::LinearizeArrayDimsCpp`.

LLAMA also provides a helper :cpp:`llama::mapping::maxLanes` which can be used to determine the maximum vector lanes which can be used for a given record dimension and vector register size.
In this example, the inner array a size of N so even the largest type in the record dimension can fit N times into a vector register of 256bits size (e.g. AVX2).

.. code-block:: C++

    llama::mapping::AoSoA<ArrayDims, RecordDim,
        llama::mapping::maxLanes<RecordDim, 256>> mapping{arrayDimsSize};


One mapping
-----------

The One mapping is intended to map all coordinates in the array dimensions onto the same memory location.
This is commonly used in  the `llama::One` virtual record, but also offers interesting applications in conjunction with the `llama::mapping::Split` mapping.


Split mapping
-------------

WARNING: This is an experimental feature and might completely change in the future.

The Split mapping is a meta mapping in the sense, that it transforms the record dimension and delegates mapping to other mappings.
Using a record coordinate, a subtree of the record dimension is selected and mapped using one mapping.
The remaining record dimension is mapped using a second mapping.

.. code-block:: C++

    llama::mapping::Split<ArrayDims, RecordDim,
        llama::RecordCoord<1>, llama::mapping::SoA, llama::mapping::PackedAoS>
            mapping{arrayDimsSize}; // maps the subtree at index 1 as SoA, the rest as packed AoS

Split mappings can be nested to map a record dimension into even fancier combinations.


.. _label-tree-mapping:

Tree mapping
------------------

WARNING: The tree mapping is currently not maintained and we consider deprecation.

The LLAMA tree mapping is one approach to archieve the goal of mixing different mapping approaches.
Furthermore, it tries to establish a general mapping description language and mapping definition framework.
Let's take the example record dimension from the :ref:`dimensions section<label-dimensions>`:

.. image:: ../images/layout_tree.svg

As already mentioned this is a compile time tree. The idea of the tree mapping
is now to extend this model to a compile time tree with run time annotations
representing the repetition of branches and to define tree operations which
create new trees out of the old ones while providing methods to translate tree
coordinates from one tree to another.

This is best demonstrated by an example.
First of all the array dimensions needs to be represented as such an tree too.
Let's assume array dimensions of :math:`128 \times 64`:

.. image:: ../images/ud_tree_2.svg

The record dimension is already a tree, but as it has no run time influence, only
:math:`1` is annotated for these tree nodes:

.. image:: ../images/layout_tree_2.svg

Now the two trees are connected so that we can represent array and record
dimensions with one tree:

.. image:: ../images/start_tree_2.svg

The mapping works now in this way that the tree is "flattened" from left to
right using a breadth first traversal. Annotations represent repetitions of the node
branches. So for this tree we would copy the record dimension :math:`64` times and
:math:`128` times again -- basically this results in an array of struct
approach, which is most probably not desired.

So we want to transform the tree before flattening it. A struct of array
approach may look like this:

.. image:: ../images/soa_tree_2.svg

Struct of array but with a padding after each 1024 elements may look like this:

.. image:: ../images/padding_tree_2.svg

The size of the leaf type in "pad" of course needs to be determined based on the
desired aligment and sub tree sizes.

Such a tree (with smaller array dimensions for easier drawing) …

.. image:: ../images/example_tree.svg

… may look like this mapped to memory:

.. image:: ../images/example_mapping.svg

In code a tree mapping is defined as :cpp:`llama::mapping::tree::Mapping`, but
takes one more template parameter for the type of a tuple of tree operations and
a further constructor parameter for the instantiation of this tuple.

.. code-block:: C++

    auto treeOperationList = llama::Tuple{
        llama::mapping::tree::functor::LeafOnlyRT()
    };

    using Mapping = llama::mapping::tree::Mapping<
        ArrayDims,
        RecordDim,
        decltype(treeOperationList)
    >;

    Mapping mapping(
        arrayDimsSize,
        treeOperationList
    );

    // or using CTAD and an unused argument for the record dimension:
    llama::mapping::tree::Mapping mapping(
        arrayDimsSize,
        llama::Tuple{
            llama::mapping::tree::functor::LeafOnlyRT()
        },
        RecordDim{}
    );

The following tree operations are defined:

Idem
^^^^
:cpp:`llama::mapping::tree::functor::Idem` does not change the tree at all.
Basically a test functor for testing, how much the number of tree operations
has an influence on the run time.

LeafOnlyRT
^^^^^^^^^^^
:cpp:`llama::mapping::tree::functor::LeafOnlyRT` moves all run time parts of
the tree to the leaves, basically creates a struct of array as seen above.
However unlike :cpp:`llama::mapping::SoA` a combination with other mapping would
be possible.

MoveRTDown
^^^^^^^^^^
:cpp:`llama::mapping::tree::functor::MoveRTDown` moves a runtime multiplier from a node identified by a tree coordinate one level downward.
This effectively divides the annotation at the node by a given factor and multiplies the direct child nodes by this factor.

MoveRTDownFixed
^^^^^^^^^^^^^^^
Same as MoveRTDown but with a compile time factor.

Dump visualizations
-------------------

Sometimes it is hard to image how data will be laid out in memory by a mapping.
LLAMA can create a grafical representation of a mapping instance as SVG image or HTML document:

.. code-block:: C++

    #include <llama/DumpMapping.hpp>

    std::ofstream{filename + ".svg" } << llama::toSvg (mapping);
    std::ofstream{filename + ".html"} << llama::toHtml(mapping);

Since this feature is not often needed, it currently resides in a separate header :cpp:`llama/DumpMapping.hpp`
and is not included as part of :cpp:`llama.hpp`.

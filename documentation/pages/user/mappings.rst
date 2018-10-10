.. include:: ../common.rst

.. _label-mappings:

Mappings
========

One of the basic tasks of LLAMA is to map an address in the user domain and
datum domain to some address in allocated memory space. This isn't an easy task,
especially not if the compiler shall still be able to optimize the resulting
memory accesses (vectorization, reodering, aligned load, etc.). The compiler
needs to *understand* the semantic of the mapping at compile time. Otherwise it
is impossible to archieve a good performance.

LLAMA mapping interface
-----------------------

The LLAMA mapping interface is quite simple and is explained in detail in the
:ref:`Factory API section <label-api-factory>`. As user you only have to
know that there is a simple interface every mapping has to be implemented
against.

All mapping have an interface which binds the user domain and datum domain type
like this

.. code-block:: C++

    using Mapping = llama::mapping::SomeMapping<
        UserDomain,
        DatumDomain
    >;

This mapping needs to be ininstantiated with the run time user domain size and
an optional mapping specific parameter:

.. code-block:: C++

    Mapping mapping(
      userDomainSize
      // ,optional parameter
    );

Afterwards it can be used for the :ref:`factory <label-factory>`.

It is possible to directly realize simple mappings such as array of struct,
struct of array or padding for this interface. However a connecting or mixing
of these mappings is not possible. To address this, mappings themself can define
some kind of mapping language which itself can archieve such goals.

Which approach of a mapping language is the best, is active research. The tree
mapping is one attempt of an universal mapping language suitable for many
architectures. However even if it points out that this approach is not working
well, it is trivial to switch to a new mapping method without changing the whole
code as the mapping is totally independent of the other parts of LLAMA.

Native mappings
^^^^^^^^^^^^^^^

If only array of struct or struct of array is needed, the LLAMA provides two
native mappings which show a good performance for all tested compilers (gcc,
clang, cuda, intel):

.. code-block:: C++

    llama::mapping::SoA

.. code-block:: C++

    llama::mapping::AoS

However as stated it is not possible to combine these
mappings with padding, blocking or some other desired more complex mappings.

.. _label-tree-mapping:

LLAMA tree mapping
^^^^^^^^^^^^^^^^^^

The LLAMA tree mapping is one approach to archieve the goal of mixing differnt
mappings approaches. Let's e.g. take the example datum domain from the
:ref:`domain section<label-domains>`:

.. only:: html

  .. image:: ../../images/layout_tree.svg

.. only:: latex

  .. image:: ../../images/layout_tree.pdf

As already mentioned this is a compile time tree. The idea of the tree mapping
is now to extend this modell to a compile time tree with run time annotations
representing the repetition of branches and to define tree operations which
create new trees of the old ones while providing methods to translate tree
coordinates from one tree to another.

Best is to see this by an example. First of all the user domain needs to be
represented as such an annotated tree, too. Let's assume an user domain of
:math:`128 \times 64`:

.. only:: html

  .. image:: ../../images/ud_tree_2.svg

.. only:: latex

  .. image:: ../../images/ud_tree_2.pdf

As the datum domain has no run time influence, only :math:`1` is annotated for
these tree nodes:

.. only:: html

  .. image:: ../../images/layout_tree_2.svg

.. only:: latex

  .. image:: ../../images/layout_tree_2.pdf

Now the two trees are connected so that we can represent user domain and datum
domain with one tree:

.. only:: html

  .. image:: ../../images/start_tree_2.svg

.. only:: latex

  .. image:: ../../images/start_tree_2.pdf

The mapping works now in this way that the tree is "flattened" from left to
right. Keep in mind that the annotation represent repetitions of the node
branches. So for this tree we would copy the datum domain :math:`64` times and
:math:`128` times again -- basically this is an array of struct approach, which
is most probably not desired.

So we need to transform the tree before flattening it. The struct of array
approach may look like this:


.. only:: html

  .. image:: ../../images/soa_tree_2.svg

.. only:: latex

  .. image:: ../../images/soa_tree_2.pdf

Struct of array, but with a padding after each 1024 elements may look like this:

.. only:: html

  .. image:: ../../images/padding_tree_2.svg

.. only:: latex

  .. image:: ../../images/padding_tree_2.pdf

The size of the "type" *pad* of course needs to be determined based on the
desired aligment and sub trees sizes.

Such a tree (with smaller user domain for easier drawing) …

.. only:: html

  .. image:: ../../images/example_tree.svg

.. only:: latex

  .. image:: ../../images/example_tree.pdf

… may look like this mapped to memory:

.. only:: html

  .. image:: ../../images/example_mapping.svg

.. only:: latex

  .. image:: ../../images/example_mapping.pdf

The tree mapping is defined as :cpp:`llama::mapping::tree::Mapping`, but takes
one more template parameter for the type of a list the tree operations and a
further constructor parameter for the instantiation of this list.

.. code-block:: C++

        auto treeOperationList = llama::makeTuple(
            llama::mapping::tree::functor::LeafOnlyRT( )
        );

        using Mapping = llama::mapping::tree::Mapping<
            UserDomain,
            DatumDomain,
            decltype( treeOperationList )
        >;

        Mapping const mapping(
            userDomainSize,
            treeOperationList
        );

The following tree operations are defined yet.

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

More operations will follow in the future, as said this approach is active
reasearch and may even change in the near future.

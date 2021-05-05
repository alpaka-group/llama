.. include:: common.rst

.. _label-view:

Copying between views
=====================

Especially when working with hardware accelerators such as GPUs or offloading to many-core processors, explicit copy operations call for memory chunks as big as possible to reach good throughput.

Copying the contents of a view from one memory region to another if mapping and size are identical is trivial.
However, if the mapping differs, a direct copy of the underlying memory is wrong.
In most cases only field-wise copy operations are possible.

There is a small class of remaining cases where the mapping is the same, but the size or shape of the view is different, or the record dimension differ slightly, or the mappings are very related to each other.
E.g. when both mappings use SoA, but one time with, one time without padding, or a specific field is missing on one side.
Or two AoSoA mappings with a different inner array length.
In those cases an optimized copy procedure is possible, copying larger chunks than mere fields.

Practically, it is hard to figure out the biggest possible memory chunks to copy at compile time, since the mappings can always depend on run time parameters.
E.g. a mapping could implement SoA if the view is bigger than 255 records, but use AoS for a smaller size.

Three solutions exist for this problem:

1. Implement specializations for specific combinations of mappings, which reflect the properties of these.
This is relevant if an application uses a set of similar mappings and the copy operation between them is the bottle neck.
However, for every new mapping a new specialization is needed.

2. A run time analysis of the two views to find contiguous memory chunks.
The overhead is probably big, especially if no contiguous memory chunks are identified.

3. A compile time analysis of the mapping function.
This requires the mapping to be formulated in a way which is fully consumable via constexpr and template meta programming, probably at the cost of read- and maintainability.

An additional challenge comes from copies between different address spaces where elementary copy operations require calls to external APIs which profit especially from large chunk sizes.
In that case it may make sense to use a smaller intermediate view to shuffle a chunk from one mapping to the other inside the same address space and then perform a copy of that chunk into the other address space.
This shuffle could be performed in the source or destination address space and potentially overlap with shuffles and copies of other chunks in an asynchronous workflow.

The `async copy example <https://github.com/alpaka-group/llama/blob/master/examples/asynccopy/asynccopy.cpp>`_ tries to show an asynchronous copy/shuffle/compute workflow.
This example applies a bluring kernel to an RGB-image, but also may work only on two or one channel instead of all three.
Not used channels are not allocated and especially not copied.

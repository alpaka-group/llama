.. include:: common.rst

.. _label-view:

Copying between views
=====================

Especially when working with hardware accelerators such as GPUs, or offloading to many-core processors, explicit copy operations call for large, contiguous memory chunks to reach good throughput.

Copying the contents of a view from one memory region to another if mapping and size are identical is trivial.
However, if the mapping differs, a direct copy of the underlying memory is wrong.
In many cases only field-wise copy operations are possible.

There is a small class of remaining cases where the mapping is the same, but the size or shape of the view is different, or the record dimension differ slightly, or the mappings are very related to each other.
E.g. when both mappings use SoA, but one time with, one time without padding, or a specific field is missing on one side.
Or two AoSoA mappings with a different inner array length.
In those cases an optimized copy procedure is possible, copying larger chunks than mere fields.

Four solutions exist for this problem:

1. Implement specializations for specific combinations of mappings, which reflect the properties of these.
However, for every new mapping a new specialization is needed.

2. A run time analysis of the two views to find contiguous memory chunks.
The overhead is probably big, especially if no contiguous memory chunks are identified.

3. A black box compile time analysis of the mapping function.
All current LLAMA mappings are :cpp:`constexpr` and can thus be run at compile time.
This would allow to observe a mappings behavior from exhaustive sampling of the mapping function at compile time.

4. A white box compile time analysis of the mapping function.
This requires the mapping to be formulated transparently in a way which is fully consumable via meta-programming, probably at the cost of read- and maintainability.
Potentially upcoming C++ features in the area of statement reflection could improve these a lot.

Copies between different address spaces, where elementary copy operations require calls to external APIs, pose an additional challenge and profit especially from large chunk sizes.
A good approach could use smaller intermediate views to shuffle a chunk from one mapping to the other and then perform a copy of that chunk into the other address space, potentially overlapping shuffles and copies in an asynchronous workflow.

The `async copy example <https://github.com/alpaka-group/llama/blob/master/examples/asynccopy/asynccopy.cpp>`_ tries to show an asynchronous copy/shuffle/compute workflow.
This example applies a blurring kernel to an RGB-image, but also may work only on two or one channel instead of all three.
Not used channels are not allocated and especially not copied.


For the moment, LLAMA implements a generic, field-wise copy algorithm with faster specializations for combinations of SoA and AoSoA mappings.

.. code-block:: C++

    auto srcView = llama::allocView(srcMapping);
    auto dstView = llama::allocView(dstMapping);
    llama::copy(srcView, dstView); // use best copy strategy

Internally, :cpp:`llama::copy` will choose a copy strategy depending on the source and destination mapping.
This choice is done via template specializations of the :cpp:`llama::Copy` class template.
Users can add specializations of :cpp:`llama::Copy` to provide additional copy strategies:

.. code-block:: C++

    // provide special copy from AoS -> UserDefinedMapping
    template <typename ArrayExtents, typename RecordDim, bool Aligned, typename LinearizeArrayDims>
    struct Copy<
        llama::mapping::AoS<ArrayExtents, RecordDim, Aligned, LinearizeArrayDims>,
        UserDefinedMapping<ArrayExtents, RecordDim>>
    {
        template <typename SrcBlob, typename DstBlob>
        void operator()(
            const View<mapping::AoS<ArrayExtents, RecordDim, Aligned, LinearizeArrayDims>, SrcBlob>& srcView,
            View<mapping::SoA<ArrayExtents, RecordDim, DstSeparateBuffers, LinearizeArrayDims>, DstBlob>& dstView,
            std::size_t threadId, std::size_t threadCount) {
            ...
        }
    };

    llama::copy(srcView, dstView); // can delegate to above specialization now

LLAMA also allows direct access to its two copy implementations, which is mainly used for benchmarking them:

.. code-block:: C++

    llama::fieldWiseCopy(srcView, dstView); // explicit field-wise copy
    llama::aosoaCommonBlockCopy(srcView, dstView); // explicit SoA/AoSoA copy
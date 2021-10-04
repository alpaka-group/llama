.. include:: common.rst

.. _label-blobs:

Blobs
=====

When a :ref:`view <label-view>` is created, it needs to be given an array of blobs.
A blob is an object representing a contiguous region of memory where each byte is accessible using the subscript operator.
The number of blobs and the alignment/size of each blob is a property determined by the mapping used by the view.
All this is handled by :cpp:`llama::allocView()`, but I needs to be given a blob allocator to handle the actual allocation of each blob.

.. code-block:: C++

    auto blobAllocator = ...;
    auto view = llama::allocView(mapping, blobAllocator);


Every time a view is copied, it's array of blobs is copied too.
Depending on the type of blobs used, this can have different effects.
If e.g. :cpp:`std::vector<std::byte>` is used, the full storage will be copied.
Contrary, if a :cpp:`std::shared_ptr<std::byte[]>` is used, the storage is shared between each copy of the view.

.. _label-bloballocators:

Blob allocators
---------------

A blob allocator is a callable which returns an appropriately sized blob given a desired compile-time alignment and runtime allocation size in bytes.
Choosing the right compile-time alignment has implications on the read/write speed on some CPU architectures and may even lead to CPU exceptions if data is not properly aligned.
A blob allocator is called like this:

.. code-block:: C++

    auto blobAllocator = ...;
    auto blob = blobAllocator(std::integral_constant<std::size_t, Alignment>{}, size);

There is a number of a built-in blob allocators:

Shared memory
^^^^^^^^^^^^^

:cpp:`llama::bloballoc::SharedPtr` is a blob allocator creating blobs of type :cpp:`std::shared_ptr<std::byte[]>`.
These blobs will be shared between each copy of the view and only destroyed then the last view is destroyed.

Vector
^^^^^^

:cpp:`llama::bloballoc::Vector` is a blob allocator creating blobs of type :cpp:`std::vector<std::byte>`.
This means every time a view is copied, the whole memory is copied too.
When the view is moved, no extra allocation or copy operation happens.

Stack
^^^^^

When working with small amounts of memory or temporary views created frequently, it is usually beneficial to store the data directly inside the view, avoiding a heap allocation.

:cpp:`llama::bloballoc::Stack` addresses this issue and creates blobs of type :cpp:`llama::Array<std::byte, N>`, where :cpp:`N` is a compile time value passed to the allocator.
These blobs are copied every time their view is copied.
:cpp:`llama::One` uses this facility.
In many such cases, the extents of the array dimensions are also known at compile time, so they can be specified in the template argument list of :cpp:`llama::ArrayExtents`.

Creating a small view of :math:`4 \times 4` may look like this:

.. code-block:: C++

    using ArrayExtents = llama::ArrayExtents<4, 4>;
    constexpr ArrayExtents extents{};

    using Mapping = /* a simple mapping */;
    auto blobAllocator = llama::bloballoc::Stack<
        extents[0] * extents[1] * llama::sizeOf<RecordDim>::value
    >;
    auto miniView = llama::allocView(Mapping{extents}, blobAllocator);

    // or in case the mapping is constexpr and produces just 1 blob:
    constexpr auto mapping = Mapping{extents};
    auto miniView = llama::allocView(mapping, llama::bloballoc::Stack<mapping.blobSize(0)>{});

For :math:`N`-dimensional one-record views a shortcut exists, returning a view with just one record on the stack:

.. code-block:: C++

    auto tempView = llama::allocViewStack<N, RecordDim>();


Non-owning blobs
----------------

If a view is needed based on already allocated memory, the view can also be directly constructed with an array of blobs,
e.g. an array of :cpp:`std::byte*` pointers or :cpp:`std::span<std::byte>` to the existing memory regions.
Everything works here as long as it can be subscripted by the view like :cpp:`blob[offset]`.
One needs to be careful though, since now the ownership of the blob is decoupled from the view.
It is the responsibility of the user now to ensure that the blobs outlive the views based on them.

Alpaka
^^^^^^

The following descriptions are for alpaka users.
Without an understanding of alpaka they may hard to understand.

LLAMA features some examples using `alpaka <https://github.com/alpaka-group/alpaka>`_ for the abstraction of computation parallelization.
Alpaka has its own memory allocation functions for different memory regions (e.g. host, device and shared memory).
Additionally there are some cuda-inherited rules which make e.g. sharing memory regions hard (e.g. no possibility to use a :cpp:`std::shared_ptr` on a GPU).

Alpaka creates and manages memory using buffers.
However, a pointer to the underlying storage can be obtained, which may be used for a view:

.. code-block:: C++

    auto buffer = alpaka::allocBuf<std::byte, std::size_t>(dev, size);
    auto view = llama::View<Mapping, std::byte*>{mapping, {alpaka::getPtrNative(buffer)}};

Shared memory is created by alpaka using a special function returning a reference to a shared variable.
To allocate storage for LLAMA, we can allocate a shared byte array using alpaka and then pass the address of the first element to a LLAMA view.

.. code-block:: C++

    auto& sharedMem = alpaka::declareSharedVar<std::byte[sharedMemSize], __COUNTER__>(acc);
    auto view = llama::View<Mapping, std::byte*>{mapping, {&sharedMem[0]}};

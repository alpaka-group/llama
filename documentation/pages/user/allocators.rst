.. include:: ../common.rst

.. _label-allocators:

Allocator
=========

To create a :ref:`view <label-view>` the :ref:`factory <label-factory>` needs to
allocate memory. This allocator is explicit given to the factory and has the
only task to return an object which the view can bytewise address.

Concept
-------

It depends on the chosen allocator whether the created view has its own memory
which will be copied and freed when needed, whether it is shared or has no
ownership information at all. The last apprach is important for reusing already
existing memory of third party libraries as seen in the
:ref:`respective subsection <label-allocators-third-party>`.

Builtin
-------

The implementation of the shipped allocators is not important for the end user
(see :ref:`API <label-api-allocators>` for details) but the behaviour while
beeing copied.

Shared memory
^^^^^^^^^^^^^

:cpp:`llama::allocator::SharedPtr` is the most easy and fastest allocator to use
as the memory of the view is never copied, but a global, shared reference
counter keeps track how many copies exist using :cpp:`std::shared_ptr` in the
background. If the last copy of a view runs out of scope the memory is freed.

Furthermore a (compile time) alignment option can be given to the allocator to
improve memory read and write:

.. code-block:: C++

    llama::allocator::SharedPtr< 256 >

Vector
^^^^^^

:cpp:`llama::allocator::Vector` uses :cpp:`std::vector<unsigned char>` in the
background. This means every time a view is copied, the whole memory is copied
too. When the view is moved, no extra allocation or copy operation happens.
Just as the allocator above, it is possible to give the memory alignment as
template parameter:

.. code-block:: C++

    llama::allocator::Vector< 256 >

Stack
^^^^^

When creating views for the whole application data, it make sense to allocate
big chunks of memory. However when working with smaller amounts of memory or
having some temporary views which need to be recreated a lot, heap allocations
are inefficient and it makes more sense to store the data directly on the stack.

:cpp:`llama::allocator::Stack` addresses this challenge. The memory object
returned from this allocator holds the needed memory in an array on the stack in
itself. However at the moment one disadvantage is, that the size of the stack
memory needs to be passed as template parameter. It is possible to determine the
needed space at compile time for simple mappings, but e.g. padding needs a
run time chooseable size. This will be addressed in the future.

Getting a small view of :math:`4 \times 4` may look like this:

.. code-block:: C++

    using UserDomain = llama::UserDomain< 2 >;
    constexpr UserDomain miniSize{ 4, 4 };

    using MiniMapping = /* some simple mapping */;

    auto miniView = llama::Factory<
        MiniMapping,
        llama::allocator::Stack<
            miniSize[0] * miniSize[1] * llama::SizeOf< /* some datum domain */ >::value
        >
    >::allocView( MiniMapping( miniSize ) );

For :math:`N`-dimensional one-element views a shortcut exists returning a view
with just one element without any padding, aligment, or whatever on the stack:

.. code-block:: C++

    auto tempView = llama::tempAlloc< N, /* some datum domain */ >();

.. _label-allocators-third-party:

Third party
-----------

The allocators are not only important for giving the views fine-tuned control
over allocation and copying overhead, but also an interface for other third
party libraries. For use with these, the
:ref:`Factory API section <label-api-factory>` documents the interface
needed to implement against.

.. _label-allocators-alpaka:

Alpaka
^^^^^^

However LLAMA feature some examples using
`alpaka <https://github.com/ComputationalRadiationPhysics/alpaka>`_
for the abstraction of computation parallelization. Alpaka has not only its own
memory allocation functions for different memory regions (e.g. host, device and
shared memory) but also some cuda-inherited rules which make e.g. sharing
memory regions hard (e.g. no possibility to use a :cpp:`std::shared_ptr` on a
GPU).

Although those allocators are no official part of LLAMA, they will be described
here nevertheless as they are
`shipped with the examles <https://github.com/ComputationalRadiationPhysics/llama/blob/master/examples/common/AlpakaAllocator.hpp>`_
and as alpaka is (not only namewise) one of the most useful third party
libraries LLAMA works together with. All examples are under a free license
without copyleft and may be freely copied and altered as needed.

The following descriptions are for alpaka and LLAMA users. Without an
understanding of alpaka they may hard to understand and can probably be skipped.

Alpaka
""""""
:cpp:`common::allocator::Alpaka` creates a buffer for a given device. Internally
the well known alpaka buffer object is used.

.. code-block:: C++

    common::allocator::Alpaka<
        DevAcc, // some alpaka device type
        Size,   // some size type, e.g. std::size_t
    >

As usual with alpaka the size type is an explicit parameter. If the view is
copied, the internal alpaka buffer is copied, too, which internally uses a
:cpp:`std::shared_ptr` just as the shared memory allocator above.

AlpakaMirror
""""""""""""
As probably known it is not possible to directly give an alpaka buffer as a
kernel argument -- at least not for kernels running on Nvidia GPUs. Taking the
native device pointer is an often seen solution, but this is exactly what LLAMA
tries to avoid.

So with :cpp:`common::allocator::AlpakaMirror` a view can be created, which
maps the already allocated memory of another LLAMA mapping.

.. code-block:: C++

    using Mapping = /* some mapping */;
    Mapping mapping( /* parameters */ );

    auto view = llama::Factory<
        Mapping,
        common::allocator::Alpaka<
            DevAcc,
            Size
        >
    >::allocView( mapping, devAcc );

    auto mirror = llama::Factory<
        Mapping,
        common::allocator::AlpakaMirror<
            DevAcc,
            Size,
            Mapping
        >
    >::allocView( mapping, view );

This :cpp:`mirror` view is ready to be copied to any offload device.

AlpakaShared
""""""""""""
Allocation of shared memory is special in cuda and even more special in the
C++ abstraction of alpaka. So an own allocator for this special memory is
defined. Two different kinds of shared memory exist: dynamically and statically
allocated. Right now only an allocator for static shared memory exists. Like for
:cpp:`llama::allocator::Stack` the size of the memory region needs to be known
at compile time.

.. code-block:: C++

    using UserDomain = llama::UserDomain< 2 >;
    constexpr UserDomain sharedSize{ 16, 16 };

    using Mapping = /* some mapping */;
    Mapping mapping( /* parameters */ );

    auto view = llama::Factory<
        Mapping,
        common::allocator::AlpakaShared<
            DevAcc,
            sharedSize[0] * sharedSize[1] * llama::SizeOf< /* some datum domain */ >::value
            __COUNTER__
        >
    >::allocView( mapping, devAcc );

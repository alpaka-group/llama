Introduction
============

Motivation
----------

Current hardware architectures are heterogeneous and it seems they will get even more heterogeneous in the future.
A central challenge of today's software development is portability between theses hardware architectures without leaving performance on the table.
This often requires separate code paths depending on the target system.
But even then, sometimes projects last for decades while new architectures rise and fall, making it is dangerous to settle for a specific data structure.

Performance portable parallelism to exhaust multi-, manycore and GPU hardware is addressed in recent developments like
`alpaka <https://github.com/alpaka-group/alpaka>`_ or
`Kokkos <https://github.com/kokkos/kokkos>`_.

However, efficient use of a system's memory and cache hierarchies is crucial as well and equally heterogeneous.
General solutions or frameworks seem not to exist yet.
First attempts are AoS/SoA container libraries like
`SoAx <https://www.sciencedirect.com/science/article/pii/S0010465517303983>`_ or 
`Intel's SDLT <https://software.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/libraries/introduction-to-the-simd-data-layout-templates.html>`_),
Kokkos's views or the proposed `std::mdspan <http://wg21.link/p0009r10>`_).

Let's consider an example.
It is well-known that accessing complex data in a struct of array (SoA) manner is most of the times faster than array of structs (AoS):

.. code-block:: C++

    // Array of Struct   |   // Struct of Array
    struct               |   struct
    {                    |   {
        float r, g, b;   |       float r[64][64], g[64][64], b[64][64];
        char a;          |       char a[64][64];
    } image[64][64];     |   } image;

Even this small decision between SoA and AoS has a quite different access style in code,
:cpp:`image[x][y].r` vs. :cpp:`image.r[x][y]`.
So the choice of layout already is usually quite infectious on the code we use to access a data structure.
For this specific example, research and ready to use libraries already exist.

But there are more useful mappings than SoA and AoS, such as:

 * blocking of memory (like partly using SoA inside an AoS approach)
 * strided access of data (e.g. odd indexes after each other)
 * padding
 * separating frequently accessed data from the rest (hot/cold data separation)
 * ...

Moreover, software is often using various heterogenous memory architectures such as RAM, VRAM, caches, memory-mapped devices or files, etc.
A data layout optimized for a specific CPU may be inefficient on a GPU or only slowly transferable over network.
A single layout -- not optimal for each architecture -- is very often a trade-off.
An optimal layout is highly dependent on the architecture, the scaling of the problem and of course the chosen algorithm.

Furthermore, third party libraries may expect specific memory layouts at their interface, into which custom data structures need to be converted.

Goals
-----

LLAMA tries to achieve the following goals:

* Allow users to express a generic data structure independently of how it is stored.
  Consequently, algorithms written against this data structure’s interface are not bound to the data structure’s layout in memory.
  This requires a data layout independent way to access the data structure.
* Provide generic facilities to map the user-defined data structure into a performant data layout.
  Also allowing specialization of this mapping for specific data structures by the user.
  A data structure’s mapping is set and resolved statically at compile time, thus guaranteeing the same performance as manually written versions of a data structure.
* Enable efficient, high throughput copying between different data layouts of the same data structure, which is a necessity in heterogeneous systems.
  This requires meta data on the data layout.
  Deep copies are the focus, although LLAMA should include the possibility for zero copies and in-situ transformation of data layouts.
  Similar strategies could be adopted for message passing and copies between file systems and memory.
  (WIP)
* To be compatible with many architectures, softwares, compilers and third party libraries, LLAMA tries to stay within C++17.
  No separate description files or language is used.
* LLAMA should work well with auto vectorization approaches of modern compilers, but also support explicit vectorization on top of LLAMA.


Library overview
----------------

The following diagram gives an overview over the components of LLAMA:

.. image:: ../images/overview.svg

The core data structure of LLAMA is the :ref:`View <label-view>`,
which holds the memory for the data and provides methods to access the data.
In order to create a view, a `Mapping` is needed which is an abstract concept.
LLAMA offers many kinds of mappings and users can also provide their own mappings.
Mappings are constructed from a :ref:`Datum domain <label-dd>`, containing tags, and an :ref:`Array domain <label-ad>`.
In addition to a mapping defining the memory layout, an array of :ref:`Blobs <label-blobs>` is needed for a view, supplying the actual storage behind the view.
A blob is any object representing a contiguous chunk of memory, byte-wise addressable using :cpp:`operator[]`.
A suitable Blob array is either directly provided by the user or built using a :ref:`BlobAllocator <label-bloballocators>` when a view is created by a call to `allocView`.
A blob allocator is again an abstract concept and any object returning a blob of a requested size when calling :cpp:`operator()`.
LLAMA comes with a set of predefined blob allocators and users can again provider their own.

Once a view is created, the user can navigate on the data managed by the view.
On top of a view, a :ref:`VirtualView <label-virtualview>` can be created, offering access to a subrange of the array domain.
Elements of the array domain, called datums, are accessed on both, View and VirtualView, by calling :cpp:`operator()` with an instance of the array domain.
This access returns a :ref:`VirtualDatum <label-virtualdatum>`, allowing further access using the tags from the datum domain, until eventually a reference to actual data in memory is returned.


Example use cases
-----------------

This library is designed and written by the `software development for experiments group (EP-SFT) at CERN <https://ep-dep-sft.web.cern.ch/>`_,
by the `group for computational radiation physics (CRP) at HZDR <https://www.hzdr.de/crp>`_ and `CASUS <https://www.casus.science>`_.
While developing, we have some in house and partner applications in mind.
These example use cases are not the only targets of LLAMA, but drove the development and the feature set.

One of the major projects in EP-SFT is the `ROOT data analysis framework <https://root.cern/>`_ for data analysis in high-energy physics.
A critical component is the fast transfer of petabytes of filesystem data taken from CERN’s detectors into an efficient in-memory representation for subsequent analysis algorithms.
This data are particle interaction events, each containing a series of variable size attributes.
A typical analysis involves column selection, cuts, filters, computation of new attributes and histograms.
The data in ROOT files is stored in columnar blocks and significant effort is made to make the data flow and aggregation as optimal as possible.
LLAMA will supply the necessary memory layouts for an optimal analysis and automate the data transformations from disk into these layouts.

The CRP group works on a couple of simulation codes, e.g.
`PIConGPU <https://picongpu.hzdr.de>`_, the fastest particle in cell code
running on GPUs. Recent development efforts furthermore made the open source
project ready for other many core and even classic CPU multi core architectures
using the library alpaka. The similar
namings of alpaka and LLAMA are no coincidence. While alpaka abstracts the
parallelization of computations, LLAMA abstracts the memory access.
To get the best out of computational resources, accelerating data
structures and a mix of SoA and AoS known to perform well on GPUs is used.
The goal is to abstract these data structures with LLAMA to be able to change
them fast for different architectures.

Image processing is another big, emerging task of the group and partners. Both,
post processing of diffraction images as well as live analysis of high rate
data sources, will be needed in the near future. As with the simulation codes, the
computation devices, the image sensor data format and the problem size may vary
and a fast and easy adaption of the code is needed.

The shipped
`examples <https://github.com/alpaka-group/llama/tree/master/examples>`_
of LLAMA try to showcase the implemented feature in the intended usage.

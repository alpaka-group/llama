.. include:: common.rst

.. _label-simd:

SIMD
====

Single instruction, multiple data (SIMD) is a data parallel programming paradigm
where an operation is simultaneously performed on multiple data elements.

There is really only one goal to using SIMD and that is: performance.
SIMD improves performance by allowing the CPU to crunch more data with each instruction, thus increasing throughput.
This influenced some of the API decisions LLAMA has taken,
because there is no point in providing an API that cannot be performant.
NB: The use of SIMD technology can also improve energy efficiency, but, arguably, this also stems from improved performance.

Many hardware architectures provide dedicated instruction sets (such as AVX2 on x86, or SVE2 on ARM) to perform basic operations
such as addition, type-conversion, square-root, etc. on a vector of fundamental types (.e.g :cpp:`int` or :cpp:`float`).
Such instructions are typically accessibly in C++ via compiler intrinsic functions.

SIMD libraries
--------------

Since compiler intrinsics tend to be hard to use and inflexible (e.g. cannot just switch a code between e.g. AVX2 and AVX512),
several SIMD libraries have been developed over time.

Here is a non-exhaustive list of some active SIMD libraries we are aware of:

* `EVE <https://github.com/jfalcou/eve>`_ (C++20)
* `xsimd <https://github.com/xtensor-stack/xsimd>`_ (C++11)
* `std::simd <https://en.cppreference.com/w/cpp/experimental/simd>`_ (experimental, GCC >= 11)
* Kokkos `SIMD <https://github.com/kokkos/kokkos/tree/develop/simd>`_ (upcoming in Kokkos 3.7, used to be developed `here <https://github.com/kokkos/simd-math>`_)

SIMD interaction with LLAMA
---------------------------

SIMD is primarily a technique for expressing computations.
These computations mainly occur between registers but may have optional memory operands.
SIMD operations involving memory usually only load or store a vector of N elements from or to the memory location.
Thus, whether a code uses SIMD or not is at first glance independent of LLAMA.
The only link between SIMD programming and data layouts provided by LLAMA is transferring of N-element vectors between memory and registers instead of scalar values.

Since LLAMA's description and use of record data is rather unwieldy and lead to the creation of :cpp:`llama::One`,
a similar construct for SIMD versions of records, called :cpp:`llama::Simd`, further increases the usability of the API.

SIMD library integration with LLAMA
-----------------------------------

In order for LLAMA to make use of a third-party SIMD library,
the class template :cpp:`llama::SimdTraits` has to be specialized for the SIMD types of the SIMD library.

Here is an exemplary integration of `std::experimental::simd<T, Abi>` with LLAMA:

.. code-block:: C++

    #include <llama/llama.hpp>
    #include <experimental/simd>

    namespace stdx = std::experimental;
    template<typename T, typename Abi>
    struct llama::SimdTraits<stdx::simd<T, Abi>> {
        using value_type = T;

        inline static constexpr std::size_t lanes = stdx::simd<T, Abi>::size();

        static auto loadUnaligned(const value_type* mem) -> stdx::simd<T, Abi> {
            return {mem, stdx::element_aligned};
        }

        static void storeUnaligned(stdx::simd<T, Abi> simd, value_type* mem) {
            simd.copy_to(mem, stdx::element_aligned);
        }
    };

Each specialization :cpp:`llama::SimdTraits<Simd>` must provide:

* an alias :cpp:`value_type` to indicate the element type of the Simd.
* a :cpp:`static constexpr size_t lanes` variable holding the number of SIMD lanes of the Simd.
* a :cpp:`static auto loadUnaligned(const value_type* mem) -> Simd` function, loading a Simd from the given memory address.
* a :cpp:`static void storeUnaligned(Simd simd, value_type* mem)` function, storing the given Simd to a given memory address.

LLAMA already provides a specialization of :cpp:`llama::SimdTraits` for the built-in scalar `arithmetic types <https://en.cppreference.com/w/c/language/arithmetic_types>`_.
In that sense, these types are SIMD types from LLAMA's perspective and can be used with the SIMD API in LLAMA.

LLAMA SIMD API
--------------

SIMD codes deal with vectors of N elements.
This assumption holds as long as the code uses the same element type for all SIMD vectors.
The moment different element types are mixed, all bets are off, and various trade-offs can be made.
For this reason, LLAMA does not automatically choose a vector length and this number needs to be provided by the user.
A good idea is to query your SIMD library for a suitable size:

.. code-block:: C++

    constexpr auto N = stdx::native_simd<T>::size();

Alternatively, LLAMA provides a few constructs to find a SIMD vector length for a given record dimension:

.. code-block:: C++

    constexpr auto N1 = llama::simdLanesWithFullVectorsFor<RecordDim, stdx::native_simd>;
    constexpr auto N2 = llama::simdLanesWithLeastRegistersFor<RecordDim, stdx::native_simd>;

:cpp:`llama::simdLanesWithFullVectorsFor` ensures that the vector length is large enough
to even fully fill at least one SIMD vector of the smallest field types of the record dimension.
So, if your record dimension contains e.g. :cpp:`double`, :cpp:`int` and :cpp:`uint16_t`,
then LLAMA will choose a vector length were a :cpp:`stdx::native_simd<uint16_t>` is full.
The SIMD vectors for :cpp:`double` and :cpp:`int` would then we larger then a full vector,
so the chosen SIMD library needs to support SIMD vector lengths larger than the native length.
E.g. the :cpp:`stdx::fixed_size_simd<T, N>` type allows :cpp:`N` to be larger than the native vector size.

:cpp:`llama::simdLanesWithLeastRegistersFor` ensures that the smallest number of SIMD registers is needed
and may thus only partially fill registers for some data types.
So, given the same record dimension, LLAMA would only fill the SIMD vectors for the largest data type (:cpp:`double`).
The other SIMD vectors would only be partially filled,
so the chosen SIMD library needs to support SIMD vector lengths smaller than the native length.

After choosing the SIMD vector length,
we can allocate SIMD registers for :cpp:`N` elements of each record dimension field using :cpp:`llama::SimdN`:

.. code-block:: C++

    llama::SimdN<RecordDim, N, stdx::fixed_size_simd> s;

We expect :cpp:`llama::SimdN` to be also used in heterogeneous codes where we want to control the vector length at compile time.
A common use case would be to have a SIMD length in accord with the available instruction set on a CPU,
and a SIMD length of 1 on a GPU.
In the latter case, it is important that the code adapts itself to not make use of types from a third-party SIMD library,
as these cannot usually be compiled for GPU targets.
Therefore, for an :cpp:`N` of 1, LLAMA will not use SIMD types:

+-----------------------+-------------------------------------+
+ SimdN<T, N>           | N > 1               | N == 1        |
+----------+------------+---------------------+---------------+
+          | record dim | :cpp:`One<Simd<T>>` | :cpp:`One<T>` |
+ :cpp:`T` +------------+---------------------+---------------+
+          | scalar     | :cpp:`Simd<T>`      | :cpp:`T`      |
+----------+------------+---------------------+---------------+

Alternatively, there is also a version without an enforced SIMD vector length:

.. code-block:: C++

    llama::Simd<RecordDim, stdx::native_simd> s;

Mind however, that with :cpp:`llama::Simd`, LLAMA does not enforce a vector width.
This choice is up to the behavior of the SIMD type.
Thus, the individual SIMD vectors (one per record dimension field) may have different lengths.

:cpp:`llama::SimdN` and :cpp:`llama::Simd` both make use of the helpers :cpp:`llama::SimdizeN` and :cpp:`llama::Simdize`
to create SIMD versions of a given record dimension:

.. code-block:: C++

    using RecordDimSimdN = llama::SimdizeN<RecordDim, N, stdx::fixed_size_simd>;
    using RecordDimSimd  = llama::Simdize <RecordDim,    stdx::native_simd>;

Eventually, whatever SIMD type is built or used by the user,
LLAMA needs to be able to query its lane count in a generic context.
This is what :cpp:`llama::simdLanes` is for.

+------------------------------------+------------------------------------+
| :cpp:`T`                           | :cpp:`llama::simdLanes<T>`         |
+------------------------------------+------------------------------------+
| scalar (:cpp:`std::is_arithmetic`) | :cpp:`1`                           |
+------------------------------------+------------------------------------+
| :cpp:`llama::One<T>`               | :cpp:`1`                           |
+------------------------------------+------------------------------------+
| :cpp:`llama::SimdN<T, N, ...>`     | :cpp:`N`                           |
+------------------------------------+------------------------------------+
| :cpp:`llama::Simd<T, ...>`         | :cpp:`llama::simdLanes<First<T>>`  |
|                                    | // if equal for all fields         |
|                                    | // otherwise: compile error        |
+------------------------------------+------------------------------------+
| otherwise                          | :cpp:`llama::SimdTraits<T>::lanes` |
+------------------------------------+------------------------------------+

Use :cpp:`llama::simdLanes` in generic code which needs to handle scalars,
third-party SIMD vectors (via. :cpp:`llama::SimdTraits`, record references, :cpp:`llama::One` and LLAMA built SIMD types.

Loading and storing data between a SIMD vector and a llama view is done using :cpp:`llama::loadSimd` and :cpp:`llama::storeSimd`:

.. code-block:: C++

    llama::loadSimd(s, view(i));
    llama::storeSimd(view(i), s);

Both functions take a :cpp:`llama::Simd` and a reference into a LLAMA view as arguments.
Depending on the mapping of the view, different load/store instructions will be used.
E.g. :cpp:`llama::mapping::SoA` will allow SIMD loads/stores,
whereas :cpp:`llama::mapping::AoS` will resort to scalar loads/stores (which the compiler sometimes optimizes into SIMD gather/scatter).

Since :cpp:`llama::Simd` is a special version of :cpp:`llama::One`,
ordinary navigation to sub records and arithmetic can be performed:

.. code-block:: C++

    llama::SimdN<Vec3, N, stdx::fixed_size_simd> vel; // SIMD with N lanes holding 3 values
    llama::loadSimd(vel, view(i)(Vel{}));

    s(Pos{}) += vel; // 3 SIMD adds performed between llama::Simd vel and sub-record llama::Simd of s
    llama::storeSimd(view(i)(Pos{}), s(Pos{})); // store subpart of llama::Simd into view


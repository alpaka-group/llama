.. include:: common.rst

.. _label-view:

Iteration
=========

Array domain iterating
----------------------

The array domain spans an N-dimensional space of integral indices.
Sometimes we just want to quickly iterate over all coordinates in this index space.
This is what :cpp:`ArrayDomainRange` is for, which is a range in the C++ sense and
offers the :cpp:`begin()` and  :cpp:`end()` member functions with corresponding iterators to support STL algorithms or the range-for loop.

.. code-block:: C++

    llama::ArrayDomain<2> ad{3, 3};
    llama::ArrayDomainIndexRange range{ad};
    
    std::for_each(range.begin(), range.end(), [](auto coord) {
        // coord is {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}
    });

    for (auto coord : range) {
        // coord is {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}
    }


Datum domain iterating
----------------------

Datum domain is done using using meta programming techniques with :cpp:`llama::forEachLeaf`.
It takes a datum domain as template argument and a functor as run time parameter.
The functor is then called for each leaf of the datum domain tree with a datum coord as argument:

.. code-block:: C++

    using DatumDomain = llama::DS<
        llama::DE<x, float>,
        llama::DE<y, float>,
        llama::DE<z, llama::DS<
            llama::DE< low, short>,
            llama::DE<high, short>
        > >
    >;

    MyFunctor functor;
    llama::forEachLeaf<DatumDomain>(functor);

    // functor will be called with an instance of
    // * DatumCoord<0> for x
    // * DatumCoord<1> for y
    // * DatumCoord<2, 0> for z.low
    // * DatumCoord<2, 1> for z.high

Optionally, a subtree of the DatumDomain can be chosen.
The subtree is described either via a `DatumCoord` or a series of tags.

.. code-block:: C++

    // "functor" will be called for
    // * z.low
    // * z.high
    llama::forEachLeaf<DatumDomain>(functor, z{});

    // "functor" will be called for
    // * z.low
    llama::forEachLeaf<DatumDomain>(functor, z{}, low{});

    // "functor" will be called for
    // * z.high
    llama::forEachLeaf<DatumDomain>(functor, llama::DatumCoord<2, 1>{});

The functor type itself needs to provide the :cpp:`operator()` with one templated parameter, to which 
the coordinate of the leaf in the datum domain tree is passed.
A polymorphic lambda is recommented to be used as a functor.

.. code-block:: C++

    auto vd = view(23, 43);
    llama::forEachLeaf<DatumDomain>([&](auto coord) {
        vd(coord) = 1337.0f;
    });

    // or using a struct:

    template<typename VirtualDatum, typename Value>
    struct SetValueFunctor {
        template<typename Coord>
        void operator()(Coord coord) {
            vd(coord) = value;
        }
        VirtualDatum vd;
        const Value value;
    };

    SetValueFunctor<decltype(vd), float> functor{1337.0f};
    llama::forEachLeaf<DatumDomain>(functor);

A more detailed example can be found in the
`simpletest example <https://github.com/alpaka-group/llama/blob/master/examples/simpletest/simpletest.cpp>`_.


View iterators
--------------

TODO

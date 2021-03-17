.. include:: common.rst

.. _label-view:

Iteration
=========

Datum domain iterating
----------------------

It is trivial to iterate over the array domain, especially using :cpp:`ArrayDomainRange` and although it is done at run
time the compiler can optimize a lot e.g. with tree vectorization or loop unrolling, especially with the beforementioned macros.

It is also possible to iterate over the datum domain.
This is achieved using meta programming techniques with :cpp:`llama::forEach`.
It takes a datum domain as template argument and a functor as run time parameter.
The functor is then called for each leaf of the datum domain tree:

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

    // "functor" will be called for
    // * x
    // * y
    // * z.low
    // * z.high
    llama::forEach<DatumDomain>(functor);

Optionally, a subtree of the DatumDomain can be chosen.
The subtree is described either via a `DatumCoord` or a series of tags.

.. code-block:: C++

    // "functor" will be called for
    // * z.low
    // * z.high
    llama::forEach<DatumDomain>(functor, z{});

    // "functor" will be called for
    // * z.low
    llama::forEach<DatumDomain>(functor, z{}, low{});

    // "functor" will be called for
    // * z.high
    llama::forEach<DatumDomain>(functor, llama::DatumCoord<2, 1>{});

The functor type itself is a struct which provides the :cpp:`operator()` with one template parameter,
the coordinate of the leaf in the datum domain tree, the functor is called on.

.. code-block:: C++

    template<typename VirtualDatum, typename Value>
    struct SetValueFunctor {
        template<typename Coord>
        void operator()(Coord coord) {
            vd(coord) = value;
        }
        VirtualDatum vd;
        const Value value;
    };

    // ...

    auto vd = view(23, 43);

    SetValueFunctor<decltype(vd), float> functor{1337.0f};
    llama::forEach<DatumDomain>(functor);

    // or using a lambda function:
    llama::forEach<DatumDomain>([&](auto coord) {
        vd(coord) = value;
    });

A more detailed example can be found in the
`simpletest example <https://github.com/alpaka-group/llama/blob/master/examples/simpletest/simpletest.cpp>`_.

Array domain iterating
----------------------

TODO

View iterators
--------------

TODO

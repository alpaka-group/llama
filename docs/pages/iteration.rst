.. include:: common.rst

.. _label-view:

Iteration
=========

Array dimensions iteration
--------------------------

The array dimensions span an N-dimensional space of integral indices.
Sometimes we just want to quickly iterate over all coordinates in this index space.
This is what :cpp:`llama::ArrayDimsIndexRange` is for, which is a range in the C++ sense and
offers the :cpp:`begin()` and  :cpp:`end()` member functions with corresponding iterators to support STL algorithms or the range-for loop.

.. code-block:: C++

    llama::ArrayDims<2> ad{3, 3};
    llama::ArrayDimsIndexRange range{ad};
    
    std::for_each(range.begin(), range.end(), [](auto coord) {
        // coord is {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}
    });

    for (auto coord : range) {
        // coord is {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}
    }


Record dimension iteration
--------------------------

The record dimension is iterated using :cpp:`llama::forEachLeaf`.
It takes a record dimension as template argument and a functor as function argument.
The functor is then called for each leaf of the record dimension tree with a record coord as argument:

.. code-block:: C++

    using RecordDim = llama::Record<
        llama::Field<x, float>,
        llama::Field<y, float>,
        llama::Field<z, llama::Record<
            llama::Field< low, short>,
            llama::Field<high, short>
        > >
    >;

    MyFunctor functor;
    llama::forEachLeaf<RecordDim>(functor);

    // functor will be called with an instance of
    // * RecordCoord<0> for x
    // * RecordCoord<1> for y
    // * RecordCoord<2, 0> for z.low
    // * RecordCoord<2, 1> for z.high

Optionally, a subtree of the RecordDim can be chosen for iteration.
The subtree is selected either via a `RecordCoord` or a series of tags.

.. code-block:: C++

    // "functor" will be called for
    // * z.low
    // * z.high
    llama::forEachLeaf<RecordDim>(functor, z{});

    // "functor" will be called for
    // * z.low
    llama::forEachLeaf<RecordDim>(functor, z{}, low{});

    // "functor" will be called for
    // * z.high
    llama::forEachLeaf<RecordDim>(functor, llama::RecordCoord<2, 1>{});

The functor type itself needs to provide the :cpp:`operator()` with one templated parameter, to which 
the coordinate of the leaf in the record dimension tree is passed.
A polymorphic lambda is recommented to be used as a functor.

.. code-block:: C++

    auto vd = view(23, 43);
    llama::forEachLeaf<RecordDim>([&](auto coord) {
        vd(coord) = 1337.0f;
    });

    // or using a struct:

    template<typename VirtualRecord, typename Value>
    struct SetValueFunctor {
        template<typename Coord>
        void operator()(Coord coord) {
            vd(coord) = value;
        }
        VirtualRecord vd;
        const Value value;
    };

    SetValueFunctor<decltype(vd), float> functor{1337.0f};
    llama::forEachLeaf<RecordDim>(functor);

A more detailed example can be found in the
`simpletest example <https://github.com/alpaka-group/llama/blob/master/examples/simpletest/simpletest.cpp>`_.


View iterators
--------------

Iterators on views are useful but pose a couple of difficulties.
Therefore, only 1D iterators are supported currently.
Higher dimensional iterators are difficult to get right if we also want to preserve good codegen.
Multiple nested loops seem to be superior to a single iterator over multiple dimensions.

Having an iterator to a view opens up the standard library for use in conjunction with LLAMA:

.. code-block:: C++

    using ArrayDims = llama::ArrayDims<1>;
    // ...
    auto view = llama::allocView(mapping);

    for (auto vd : view) {
        vd(x{}) = 1.0f;
        vd(y{}) = 2.0f;
        vd(z{}, low{}) = 3;
        vd(z{}, high{}) = 4;
    }
    std::transform(begin(view), end(view), begin(view), [](auto vd) { return vd * 2; });
    const float sumY = std::accumulate(begin(view), end(view), 0, [](int acc, auto vd) { return acc + vd(y{}); });

    // C++20:

    for (auto x : view | std::views::transform([](auto vd) { return vd(x{}); }) | std::views::take(2))
        // ...

Since virtual records interact with each other based on the tags and not the underlying mappings, we can also use iterators from multiple views together:

.. code-block:: C++

    auto aosView = llama::allocView(llama::mapping::AoS<ArrayDims, RecordDim>{arrayDimsSize});
    auto soaView = llama::allocView(llama::mapping::SoA<ArrayDims, RecordDim>{arrayDimsSize});
    // ...
    std::copy(begin(aosView), end(aosView), begin(soaView));

    auto innerProduct = std::transform_reduce(begin(aosView), end(aosView), begin(soaView), llama::One<RecordDim>{});


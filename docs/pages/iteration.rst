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

    llama::ArrayDimsIndexRange range{llama::ArrayDims{3, 3}};
    
    std::for_each(range.begin(), range.end(), [](llama::ArrayDims<2> coord) {
        // coord is {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}
    });

    for (auto coord : range) {
        // coord is {0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}
    }


Record dimension iteration
--------------------------

The record dimension is iterated using :cpp:`llama::forEachLeaf`.
It takes a record dimension as template argument and a callable with a generic parameter as argument.
This function's :cpp:`operator()` is then called for each leaf of the record dimension tree with a record coord as argument.
A polymorphic lambda is recommented to be used as a functor.

.. code-block:: C++

    llama::forEachLeaf<Pixel>([&](auto coord) {
        // coord is RecordCoord <0, 0 >{}, RecordCoord <0, 1>{}, RecordCoord <0, 2>{} and RecordCoord <1>{}
    });

Optionally, a subtree of the record dimension can be chosen for iteration.
The subtree is selected either via a `RecordCoord` or a series of tags.

.. code-block:: C++

    llama::forEachLeaf<Pixel>([&](auto coord) {
        // coord is RecordCoord <0, 0 >{}, RecordCoord <0, 1>{} and RecordCoord <0, 2>{}
    }, color{});

    llama::forEachLeaf<Pixel>([&](auto coord) {
        // coord is RecordCoord <0, 1>{}
    }, color{}, g{});

A more detailed example can be found in the
`simpletest example <https://github.com/alpaka-group/llama/blob/master/examples/simpletest/simpletest.cpp>`_.


View iterators
--------------

Iterators on views of any dimension are supported and open up the standard library for use in conjunction with LLAMA:

.. code-block:: C++

    using Pixel = ...;
    using ArrayDims = llama::ArrayDims<1>;
    // ...
    auto view = llama::allocView(mapping);
    // ...

    // range for
    for (auto vd : view)
        vd(color{}, r{}) = 1.0f;

    auto view2 = llama::allocView (...); // with different mapping

    // layout changing copy
    std::copy(begin(aosView), end(aosView), begin(soaView));

    // transform into other view
    std::transform(begin(view), end(view), begin(view2), [](auto vd) { return vd(color{}) * 2; });

    // accumulate using One as accumulator and destructure result
    const auto [r, g, b] = std::accumulate(begin(view), end(view), One<RGB>{}, 
        [](auto acc, auto vd) { return acc + vd(color{}); });

    // C++20:
    for (auto x : view | std::views::transform([](auto vd) { return vd(x{}); }) | std::views::take(2))
        // ...

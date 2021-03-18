.. include:: common.rst

.. _label-view:

View
====

The view is the main data structure a LLAMA user will work with. It takes
coordinates in the array and datum domain and returns a reference to a datum
in memory which can be read from or written to. For easier use, some
useful operations such as :cpp:`+=` are overloaded to operate on all datum
elements inside the datum domain at once.

.. _label-factory:

View allocation
---------------

The factory creates the view. For this it takes the domains, a
:ref:`mapping <label-mappings>` and an optional :ref:`allocator <label-allocators>`.

.. code-block:: C++

    using Mapping = ...; // see next section about mappings
    Mapping mapping(arrayDomainSize); // see section about domains
    auto view = allocView(mapping); // optional allocator as 2nd argument

The :ref:`mapping <label-mappings>` and :ref:`allocator <label-allocators>`
will be explained later. For now, it is just
important to know that all those run time and compile time parameters come
together to create the view.

Data access
-----------

LLAMA tries to have an array of struct like interface.
When accessing an element of the view, the array part comes first, followed by tags from the datum domain.

In C++, runtime values like the array domain coordinate are normal function parameters
whereas compile time values such as the datum domain tags are usually given as template arguments.
However, compile time information can be stored in a type, instantiated as a value and then passed to a function template deducing the type again.
This trick allows to pass both, runtime and compile time values as function arguments.
E.g. instead of calling :cpp:`f<MyType>()` we can call :cpp:`f(MyType{})` and let the compiler deduce the template argument of :cpp:`f`.

This trick is used in LLAMA to specify the access to a value of a view.
An example access with the domains defined in the :ref:`domain section <label-domains>` could look like this:

.. code-block:: C++

    view(1, 2, 3)(color{}, g{}) = 1.0;

It is also possible to access the array domain with one compound argument like this:

.. code-block:: C++

    const ArrayDomain pos{1, 2, 3};
    view(pos)(color{}, g{}) = 1.0;
    // or
    view({1, 2, 3})(color{}, g{}) = 1.0;

The values :cpp:`color{}` and :cpp:`g{}` are not used and just serve as a way to specify the template arguments.
A direct call of the :cpp:`operator()` is also possible and looks like this:

.. code-block:: C++

    view(1, 2, 3).operator()<color, g>() = 1.0;

Alternatively, if the use of tag types is not desired or if the algorithm wants to iterate over the datum domain at compile time,
an adressing with integral datum coordinates is possible like this:

.. code-block:: C++

    view(1, 2, 3)(llama::DatumCoord<0, 1>{}) = 1.0; // color.g

This datum coordinates are zero-based, nested indices reflecting the nested tuple-like structure of the datum domain.

Notice that the :cpp:`operator()` is invoked twice in the last example and that an intermediate object is needed for this to work.
This object is a central data type of LLAMA called :cpp:`VirtualDatum`.

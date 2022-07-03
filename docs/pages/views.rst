.. include:: common.rst

.. _label-view:

View
====

The view is the main data structure a LLAMA user will work with.
It takes coordinates in the array and record dimensions and returns a reference to a record in memory which can be read from or written to.
For easier use, some useful operations such as :cpp:`+=` are overloaded to operate on all record fields inside the record dimension at once.

.. _label-factory:

View allocation
---------------

A view is allocated using the helper function :cpp:`allocView`, which takes a
:ref:`mapping <label-mappings>` and an optional :ref:`blob allocator <label-bloballocators>`.

.. code-block:: C++

    using Mapping = ...; // see next section about mappings
    Mapping mapping(extents); // see section about dimensions
    auto view = allocView(mapping); // optional blob allocator as 2nd argument

The :ref:`mapping <label-mappings>` and :ref:`blob allocator <label-bloballocators>`
will be explained later. For now, it is just
important to know that all those run time and compile time parameters come
together to create the view.

Data access
-----------

LLAMA tries to have an array of struct like interface.
When accessing an element of the view, the array part comes first, followed by tags from the record dimension.

In C++, runtime values like the array dimensions coordinates are normal function parameters
whereas compile time values such as the record dimension tags are usually given as template arguments.
However, compile time information can be stored in a type, instantiated as a value and then passed to a function template deducing the type again.
This trick allows to pass both, runtime and compile time values as function arguments.
E.g. instead of calling :cpp:`f<MyType>()` we can call :cpp:`f(MyType{})` and let the compiler deduce the template argument of :cpp:`f`.

This trick is used in LLAMA to specify the access to a value of a view.
An example access with the dimensions defined in the :ref:`dimensions section <label-dimensions>` could look like this:

.. code-block:: C++

    view(1, 2, 3)(color{}, g{}) = 1.0;

It is also possible to access the array dimensions with one compound argument like this:

.. code-block:: C++

    const llama::ArrayIndex pos{1, 2, 3};
    view(pos)(color{}, g{}) = 1.0;
    // or
    view({1, 2, 3})(color{}, g{}) = 1.0;

The values :cpp:`color{}` and :cpp:`g{}` are not used and just serve as a way to specify the template arguments.
Alternatively, an addressing with integral record coordinates is possible like this:

.. code-block:: C++

    view(1, 2, 3)(llama::RecordCoord<0, 1>{}) = 1.0; // color.g

These record coordinates are zero-based, nested indices reflecting the nested tuple-like structure of the record dimension.

Notice that the :cpp:`operator()` is invoked twice in the last example and that an intermediate object is needed for this to work.
This object is a :cpp:`llama::RecordRef`.

.. _label-virtualview:

VirtualView
-----------

Virtual views can be created on top of existing views, offering shifted access to a subspace of the array dimensions.

.. code-block:: C++

    auto view = ...;
    llama::VirtualView virtualView{view, {10, 20, 30}};
    virtualView(1, 2, 3)(color{}, g{}) = 1.0; // accesses record {11, 22, 33}

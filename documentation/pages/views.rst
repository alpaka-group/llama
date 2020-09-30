.. include:: common.rst

.. _label-view:

View
====

The view is the main data structure a LLAMA user will work with. It takes
coordinates in the user and datum domain and returns a reference to a datum
in memory which can be read or altered. For more easy use furthermore some
useful operations such as :cpp:`+=` are overloaded to operate on all datum
elements inside the datum domain at once.

.. _label-factory:

View allocation
---------------

The factory creates the view. For this it takes the domains, a
:ref:`mapping <label-mappings>` and an optional :ref:`allocator <label-allocators>`.

.. code-block:: C++

    using Mapping = ...; // see next section about mappings
    Mapping mapping(userDomainSize); // see section about domains
    auto view = allocView(mapping); // optional allocator as 2nd argument

The :ref:`mapping <label-mappings>` and :ref:`allocator <label-allocators>`
will be explained later, but are not of relevance at this point. It is just
important to know that all those run time and compile time parameters come
together to create the view.

Data access
-----------

As LLAMA tries to have an array of struct like interface.
When accessing an element of the view, the array part comes first and is called array domain.
The struct part comes afterwards and is called the datum domain.

In C++, runtime parameters like the array domain are normal function parameters whereas compile time parameters usually given as template arguments.
However, compile time information can be stored in a type, instantiated as a value and then passed to a function template deducing the type again.
This trick allows to pass both, runtime and compile time values as function arguments.
E.g. instead of calling :cpp:`f<3>()` we can call :cpp:`f(std::integral_constant<std::size_t, 3>{})`.
Furthermore, instead of calling :cpp:`f<MyType>()` we can call :cpp:`f(MyType{})`.

This trick is used in LLAMA to specify the access to a value of a view.
An example access with the domains defined in the :ref:`domain section <label-domains>` could look like this:

.. code-block:: C++

    view(1, 2, 3)(color{}, g{}) = 1.0;

LLAMA also provides a function with explicit template parameters:

.. code-block:: C++

    view(1, 2, 3).access<color, g>() = 1.0;

Unfortunately a direct call of the :cpp:`operator()` like :cpp:`view(1, 2, 3)<color, g>()` is not possible, it and would look like this:
:cpp:`view( 1, 2, 3 ).operator()<color, g>()`.
Thus, as an explicit call of the :cpp:`operator()` is needed anyway, LLAMA got an own function for this task.
Different algorithms have different requirements for accessing data.
E.g. it is also possible to access the array domain with one packed parameter like this:

.. code-block:: C++

    view({ 1, 2, 3 })(color{}, g{}) = 1.0;
    // or
    const ArrayDomain pos{1, 2, 3};
    view(pos)(color{}, g{}) = 1.0;

If the use of tag types is not desired (e.g. with the same algorithm working in the RGB or CYK colour space)
or if the algorithm wants to iterate over the datum domain at compile time,
also an adressing with the coordinate inside the tree is possible like this:

.. code-block:: C++

    view(1, 2, 3)(llama::DatumCoord< 0, 1 >{}) = 1.0; // color.g
    // or
    view(1, 2, 3).access<0, 1>() = 1.0; // color.g

VirtualDatum
^^^^^^^^^^^^

A careful reader might have noticed that the :cpp:`operator()` is "overloaded twice"
for accesses like :cpp:`view(1, 2, 3)( color{}, g{})` and that an intermediate object is needed for this to work.
This object exists and is not only an internal trick but a central data type of LLAMA called :cpp:`VirtualDatum`.

Resolving the array domain address returns such a :cpp:`VirtualDatum` with a bound array domain address.
This object can be thought of like a datum in the :math:`N`-dimensional array domain space,
but as the elements of this datum may not be in contiguous in memory, it is called virtual.

Nevertheless, it can be used like a real local object.
A virtual datum can be passed as an argument to a function (as seen in the
`nbody example <https://github.com/ComputationalRadiationPhysics/llama/blob/master/examples/nbody/nbody.cpp>`_
).
Furthermore, several arithmetic and logical operatores are overloaded:

.. code-block:: C++

    auto datum1 = view(1, 2, 3);
    auto datum2 = view(3, 2, 1);

    datum1 += datum2;
    datum1 *= 7.0; //for every element in the datum domain

    foobar(datum2);

    //With this somewhere else:
    template<typename VirtualDatum>
    void foobar(VirtualDatum vd)
    {
        vd = 42;
    }

The most common compount assignment operators ( :cpp:`=`, :cpp:`+=`, :cpp:`-=`, :cpp:`*=`,
:cpp:`/=`, :cpp:`%=` ) are overloaded. These operators directly write into the
corresponding view. Furthermore several arithmetic operators ( :cpp:`+`, :cpp:`-`,
:cpp:`*`, :cpp:`/`, :cpp:`%` ) are overloaded too, but they return a temporary object
on the stack. Althought this temporary value has a basic struct-mapping without padding and
probaly being not compatible to the mapping of the view at all, the compiler
will most probably be able to optimize the data accesses anyway as it has full
knowledge about the data in the stack and can cut out all temporary operations.

These operators work between two virtual datums, even if they have
different datum domains. It is even possible to work on parts of a virtual
datum. This returns a virtual datum with the first coordinates in the datum
domain bound. Every tag existing in both datum domains will be
matched and operated on. Every non-matching tag is ignored, e.g.

.. code-block:: C++

    using DD1 = llama::DS<
        llama::DS<llama::DE<pos
            llama::DE<x, float>
        >>,
        llama::DS<llama::DE<vel
            llama::DE <x, double>
        >>,
        llama::DE <x, int>
    >;

    using DD2 = llama::DS<
        llama::DS<llama::DE<pos
            llama::DE<x, double>
        >>,
        llama::DS<llama::DE<mom
            llama::DE<x, double>
        >>
    >;

    // Let assume datum1 using DD1 and datum2 using DD2.

    datum1 += datum2;
    // datum2.pos.x and only datum2.pos.x will be added to datum1.pos.x because
    // of pos.x existing in both datum domains although having different types.

    datum1(vel{}) *= datum2( mom() );
    // datum2.mom.x will be multiplied to datum2.vel.x as the first part of the
    // datum domain coord is explicit given and the same afterwards

The discussed operators are also overloaded for types other than :cpp:`VirtualDatum` as well so that
:cpp:`datum1 *= 7.0` will multiply 7 to every element in the datum domain.
This feature should be used with caution!

The comparison operators :cpp:`==`, :cpp:`!=`, :cpp:`<`, :cpp:`<=`, :cpp:`>`
and :cpp:`>=` are overloaded too and return the boolean value :cpp:`true` if
the operation is true for **all** matching elements of the two comparing virtual
datums respectively other type. Let's examine this deeper in an example:

.. code-block:: C++

    using A = llama::DS <
        llama::DE < x, float >,
        llama::DE < y, float >
    >;

    using B = llama::DS<
        llama::DE<z, double>,
        llama::DE<x, double>
    >;

    bool result;

    // Let assume a1 and a2 using A and b using B.

    a1(x{}) = 0.0f;
    a1(y{}) = 2.0f;

    a2(x{}) = 1.0f;
    a2(y{}) = 1.0f;
    //a2() = 1.0f; would do the same

    b (x{}) = 1.0f;
    b (z{}) = 2.0f;

    result = a1 < a2;
    //result is false, because a1.y > a2.y

    result = a1 > a2;
    //result is false, too, because now a1.x > a2.x

    result = a1 != a2;
    //result is true

    result = a2 == b;
    //result is true, because only the matching "x" matters

A partial addressing of a virtual datum like :cpp:`datum1(color{}) *= 7.0` is also possible.
:cpp:`datum1(color{})` itself returns a new virtual datum with the first tree coordiante (:cpp:`color`) being bound.
This enables e.g. to easily add a velocity to a position like this:

.. code-block:: C++

    using Particle = llama::DS<
        llama::DE<pos, llama::DS<
            llama::DE<x, float>,
            llama::DE<y, float>,
            llama::DE<z, float>
        >>,
        llama::DE<vel, llama::DS<
            llama::DE<x, double>,
            llama::DE<y, double>,
            llama::DE<z, double>
        >>,
    >;

    // Let datum be a virtual datum with the datum domain "Particle".

    datum(pos{}) += datum(vel{});

This is e.g. used in the
`nbody example <https://github.com/ComputationalRadiationPhysics/llama/blob/master/examples/nbody/nbody.cpp>`_
to update the particle velocity based on the distances of particles and to
update the position after one time step movement with the velocity.

Compiler steering
-----------------

Unfortunately C++ lacks some language features to express data and function
locality as well as independence of data.

The first shortcoming is what language extensions like cuda, OpenMP, OpenACC,
you name it try to solve. The second is mostly tackled by vendor specific
compiler extension. Both define new keywords and annotations to fill those gaps.
As LLAMA tries to stay independent from specific compiler vendors and
extensions, C preprocessor macros are used to define some directives only for a
sub set of compilers but with a unified interface for the user. Some macros can
even be overwritten from the outside to enable interoperability with libraries
such as alpaka.

Function locality
^^^^^^^^^^^^^^^^^

Every method which can be used on offloading devices (e.g. GPUs) uses the :cpp:`LLAMA_FN_HOST_ACC_INLINE` macro as attribute.
By default it is defined as:

.. code-block:: C++

    #ifndef LLAMA_FN_HOST_ACC_INLINE
        #define LLAMA_FN_HOST_ACC_INLINE inline
    #endif

When working with cuda it should be globally defined as something like :cpp:`__host__ __device__ inline`.
Please specify this as part of your CXX flags globally.
When LLAMA is used in conjunction with alpaka, please define it as :cpp:`ALPAKA_FN_ACC __forceinline__` (with CUDA) or :cpp:`ALPAKA_FN_ACC inline`.

Data (in)dependence
^^^^^^^^^^^^^^^^^^^

Compilers usually cannot assume that two data regions are
independent if the data is not fully visible to the compiler (e.g. a value completely lying on the stack).
One solution in C was the :cpp:`restrict` keyword which specifies that the memory pointed to by a pointer is not aliased by anything else.
However this does not work for more complex data structures containing pointers, and easily fails in other scenarios as well.
The :cpp:`restrict` keyword was therefore not added to the C++ language.

Another solution are compiler specific :cpp:`#pragma`\ s which tell the compiler that
**each** data access inside a loop can be assumed to be independent of each other.
This is handy and works with more complex data types, too.
So LLAMA provides a macro called :cpp:`LLAMA_INDEPENDENT_DATA` which can be put
in front of loops to tell the compiler that the data accesses in the
loop body are independent of each other -- and can savely be vectorized (which is the goal).

Datum domain iterating
----------------------

It is trivial to iterate over the array domain, especially using :cpp:`UserDomainRange` and although it is done at run
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
`simpletest example <https://github.com/ComputationalRadiationPhysics/llama/blob/master/examples/simpletest/simpletest.cpp>`_.

Thoughts on copies between views
--------------------------------

Especially when working with hardware accelerators such as GPUs or offloading to
many core procressors, explicit copy operations call for memory chunks as big as possible to reach good throughput performance.

It is trivial to copy a view from one memory region to another if mapping and size are identical.
However if the mapping differs, a direct copy of the underlying memory is wrong.
In most cases only elementwise copy operations will be possible as the memory patterns are not compatible.
There is a small class of remaining use cases where the mapping is the same, but the size of the view is different or mappings are
very related to each other. E.g. when both mappings use struct of array, but one time with, one time without padding.
In those cases an optimized copy operation would in *theory* be possible .
However *practically* it is very hard to figure out the biggest possible memory chunks to copy at compile time,
since the mappings can always depend on run time parameters.
E.g. a mapping could implement struct of array if the view is bigger than :math:`255` elements, but use array of struct for a smaller amount.

Three solutions exist for this challenge. One is to implement specializations
for specific combinations of mappings, which reflect the properties of those
mappings. This **can** be the way to go if the application shows significantly
better run times for slightly different mappings and the copy operation has be
shown to be the bottle neck. However this would be the very last optimization
step as for every new mapping a new specialization would be needed.

Another solution would be a run time analysis of the two views to find
contiguous memory chunks, but the overhead would be probably too big, especially
if no contiguous memory chunks could be found. At least in that case it may make
sense to use a (maybe smaller) intermediate view which connects the two worlds.

This last solution means that we have e.g. a view in memory region :math:`A`
with mapping :math:`A` and another view of the same size in memory region
:math:`B` with mapping :math:`B`. A third view in memory region :math:`A` but
with mapping :math:`B` could be used to reindex in region :math:`A` and then to
copy it as one big chunk to region :math:`B`.

When using two intermediate views in region :math:`A` and :math:`B` with the
same mapping but possibly different than in :math:`A` **and** :math:`B` the copy
problem can be split to smaller chunks of memory. It makes also sense to combine
this approach with an asynchronous workflow where reindexing, copying and
computation are overloayed as e.g. seen in the
`async copy example <https://github.com/ComputationalRadiationPhysics/llama/blob/master/examples/asynccopy/asynccopy.cpp>`_.

Another benefit is, that the creating and copying of the intermediate view can
be analyzed and optimized by the compiler (e.g. with vector operations).
Furthermore different (sub) datum domains may be used. The above mentioned
example e.g. applies a bluring kernel to an RGB-image, but may work only on
two or one channel instead of all three. Not used channels are not allocated and
especially not copied at all.

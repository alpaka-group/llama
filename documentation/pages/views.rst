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
:ref:`mapping <label-mappings>` and an :ref:`allocator <label-allocators>`.

.. code-block:: C++

    using Mapping = ...; // see next section about mappings
    Mapping mapping(userDomainSize); // see section about domains
    auto view = allocView(mapping,
        llama::allocator::SharedPtr{} // see over next section about allocators
    );

The :ref:`mapping <label-mappings>` and :ref:`allocator <label-allocators>`
will be explained later, but are not of relevance at this point. It is just
important to know that all those run time and compile time parameters come
together to create the view.

Data access
-----------

As LLAMA tries to have an array of struct like interface, *first* the array
part (here: user domain) and *secondly* the struct part (here: datum domain)
occurs while addressing data.

Normaly in C++ run time parameters like the user domain are normal function or
method parameters whereas compile time parameters are given as template
arguments. A trick to unify this, is to store the template parameter information
in a function parameter (e.g. an instance of
:cpp:`std::integral_constant< std::size_t, 3 >` instead of a :cpp:`< 3 >`
template parameter) or to use template argument deduction, e.g. instead of such
a function :cpp:`template< typename T > foobar()` a (probably unused) parameter
is added: :cpp:`template< typename T > foobar( T = T() )`. Instead of the
explicit template parameter :cpp:`foobar< int >()` the template parameter can be
hidden like this :cpp:`foobar( int(5) )`.

This is used in LLAMA, so one way to access a value in memory of a view created
with the domains defined in the :ref:`domain section <label-domains>` could be

.. code-block:: C++

    view( 1, 2, 3 )( color(), g() ) = 1.0;

Of course an explicit template parameter is possible, too, like this:

.. code-block:: C++

    view( 1, 2, 3 ).access< color, g >() = 1.0;

Unfortunately a direct call of the :cpp:`operator()` like
:cpp:`view( 1, 2, 3 )< color, g >()` is not possible and would look like this:
:cpp:`view( 1, 2, 3 ).operator()< color, g >()` instead. So as an explicit call
of the :cpp:`operator()` is needed anyway LLAMA got an own function for this
task.

Different algorithms have different requirements for accessing data, e.g. it
is also possible to access the user domain with one packed parameter like this

.. code-block:: C++

    view( { 1, 2, 3 } )( color(), g() ) = 1.0;
    // or
    UserDomain const pos{ 1, 2, 3 };
    view( pos )( color(), g() ) = 1.0;

If the naming in the datum domain is not important, may change (e.g. with the
same algorithm working in the RGB or CYK colour space) or is not available at
all (e.g. for :cpp:`DatumArray`) or if the algorithm wants to iterate over the
datum domain (at compile time of course), also an adressing with the coordinate
inside the tree is possible like this:

.. code-block:: C++

    view( 1, 2, 3 )( llama::DatumCoord< 0, 1 >() ) = 1.0; // color.g
    // or
    view( 1, 2, 3 ).access< 0, 1 >() = 1.0; // color.g

Here the version with the explicit :cpp:`access` function call is even shorter.

VirtualDatum
^^^^^^^^^^^^

It may have attracted attention that the :cpp:`operator()` is "overloaded twice"
for accesses like :cpp:`view( 1, 2, 3 )( color(), g() )` and that an
intermediate object is needed for this to work. This object exist and is not
only an internal trick but a central data type of LLAMA called
:cpp:`VirtualDatum`.

The resolving of the user domain address returns such a :cpp:`VirtualDatum` with
a bound user domain address. This object can be thought of like a datum in the
:math:`N`-dimensional user domain space, but as the elements of this datum will
most probably not be consecutive in memory, it is called virtual.

However it can be used like a real local object nevertheless, e.g. been given as
a parameter to a function (as seen in the
`nbody example <https://github.com/ComputationalRadiationPhysics/llama/blob/master/examples/nbody/nbody.cpp>`_
) and to increase this feeling some often needed operators are overloaded, too:

.. code-block:: C++

    auto datum1 = view( 1, 2, 3 );
    auto datum2 = view( 3, 2, 1 );

    datum1 += datum2;
    datum1 *= 7.0; //for every element in the datum domain

    foobar( datum2 );

    //With this somewhere else:
    template< typename T_VirtualDatum >
    foobar( T_VirtualDatum && vd )
    {
        vd = 42;
    }

The most needed inplace operators ( :cpp:`=`, :cpp:`+=`, :cpp:`-=`, :cpp:`*=`,
:cpp:`/=`, :cpp:`%=` ) are overloaded. These operators directly write into the
corresponding view. Furthermore the not-inplace operators ( :cpp:`+`, :cpp:`-`,
:cpp:`*`, :cpp:`/`, :cpp:`%` ) are overloaded too but return an temporary object
on the stack. Althought it has a basic struct-mapping without padding and
probaly being not compatible to the mapping of the view at all, the compiler
will most probably be able to optimize the data accesses anyway as it has full
knowledge about the data in the stack and can cut out all temporary operations.

These operators work between two virtual datums, even if they have
different datum domains. It is even possible to work on parts of a virtual
datum. This returns a virtual datum with the first coordinates in the datum
domain bound. Every namings existing in both datum domains will be
matched and operated on. Every not matching pair is ignored, e.g.

.. code-block:: C++

    using DD1 = llama::DS <
        llama::DS < llama::DE < pos
            llama::DE < x, float >
        > >,
        llama::DS < llama::DE < vel
            llama::DE < x, double >
        > >,
        llama::DE < x, int >
    >;

    using DD2 = llama::DS <
        llama::DS < llama::DE < pos
            llama::DE < x, double >
        > >,
        llama::DS < llama::DE < mom
            llama::DE < x, double >
        > >
    >;

    // Let assume datum1 using DD1 and datum2 using DD2.

    datum1 += datum2;
    // datum2.pos.x and only datum2.pos.x will be added to datum1.pos.x because
    // of pos.x existing in both datum domains although having different types.

    datum1( vel() ) *= datum2( mom() );
    // datum2.mom.x will be multiplied to datum2.vel.x as the first part of the
    // datum domain coord is explicit given and the same afterwards

The same operators are also overloaded for any other type so that
:cpp:`datum1 *= 7.0` will multiply 7 to every element in the datum domain.
Of course this may throw warnings about narrowing conversion. It is task of the
user to only use this if compatible.

The comparative operation :cpp:`==`, :cpp:`!=`, :cpp:`<`, :cpp:`<=`, :cpp:`>`
and :cpp:`>=` are overloaded too and return the boolean value :cpp:`true` if
the operation is true for **all** matching elements of the two comparing virtual
datums respectively other type. Let's examine this deeper in an example:

.. code-block:: C++

    using A = llama::DS <
        llama::DE < x, float >,
        llama::DE < y, float >
    >;

    using B = llama::DS <
        llama::DE < z, double >,
        llama::DE < x, double >
    >;

    bool result;

    // Let assume a1 and a2 using A and b using B.

    a1( x() ) = 0.0f;
    a1( y() ) = 2.0f;

    a2( x() ) = 1.0f;
    a2( y() ) = 1.0f;
    //a2() = 1.0f; would do the same

    b ( x() ) = 1.0f;
    b ( z() ) = 2.0f;

    result = a1 < a2;
    //result is false, because a1.y > a2.y

    result = a1 > a2;
    //result is false, too, because now a1.x > a2.x

    result = a1 != a2;
    //result is true

    result = a2 == b;
    //result is true, because only the matching "x" matters

A partly addressing of a virtual datums like :cpp:`datum1( color() ) *= 7.0`
is also possible. :cpp:`datum1( color() )` itself returns a new virtual datum
with the first tree coordiante (:cpp:`color`) being bound. This enables e.g. to
easily add a velocity to a position like this:

.. code-block:: C++

    using Particle = llama::DS <
        llama::DE < pos, llama::DS <
            llama::DE < x, float >,
            llama::DE < y, float >,
            llama::DE < z, float >
        > >,
        llama::DE < vel, llama::DS <
            llama::DE < x, double >,
            llama::DE < y, double >,
            llama::DE < z, double >
        > >,
    >;

    // Let datum be a virtual datum with the datum domain "Particle".

    datum( pos() ) += datum( vel() );

This is e.g. used in the
`nbody example <https://github.com/ComputationalRadiationPhysics/llama/blob/master/examples/nbody/nbody.cpp>`_
to update the particle velocity based on the distances of particles and to
update the position after one time step movement with the velocity.

Compiler steering
-----------------

Unfortunately C++ lacks some language features to express data and function
locality as well as dependence of data.

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

Every method which shall be able to be used on offloading devices (e.g. GPUs)
uses the :cpp:`LLAMA_FN_HOST_ACC_INLINE` macro in front. At default it is
defined as

.. code-block:: C++

    #ifndef LLAMA_FN_HOST_ACC_INLINE
        #define LLAMA_FN_HOST_ACC_INLINE inline
    #endif

but when working with cuda it may make sense to replace it with
:cpp:`__host__ __device__` before including or analogous for alpaka

.. code-block:: C++

    #include <alpaka/alpaka.hpp>
    #ifdef __CUDACC__
        #define LLAMA_FN_HOST_ACC_INLINE ALPAKA_FN_ACC __forceinline__
    #else
        #define LLAMA_FN_HOST_ACC_INLINE ALPAKA_FN_ACC inline
    #endif
    #include <llama/llama.hpp>

Data (in)dependence
^^^^^^^^^^^^^^^^^^^

Another problem is that compilers cannot assume that two data regions are
independent if the data is not laying on the stack completely. One solution
of C++ extensions was the :cpp:`restrict` keyword which tells that each pointer
parameter is independent of each other. However this does not work for more
complex data types hiding pointers -- as it is the idea with modern C++.

Another solution are loop :cpp:`#pragma`\ s which tell the compiler that
**each** data access inside this loop can be assumed to be independent of each
other if not explicitly determined otherwise, e.g.

.. code-block:: C

    int *a = /* ... */;
    int *b = /* ... */;

    #pragma GCC ivdep
    for (int i = 0; i < 16; i+=2)
    {
        int c[2];
        int *d = &c[1];

        c[0] = a[i  ];
        c[1] = a[i+1];

        b[i  ] += c[0];
        b[i+1] += d[0]; // c[1]
    }

In this example the compiler assumes now that :cpp:`a` and :cpp:`b` are
independent, but as :cpp:`c` and :cpp:`d` are defined inside the loop on the
stack the compilers "sees" that they are not independent although we put
:cpp:`#pragma GCC ivdep` before.

This is handy and works with more complex data types, too. However nobody wants
a :cpp:`#pragma` for every C++11 compiler existing in the world in front of
every loop (which needs to be updated when a new compiler directive is added).
:cpp:`#pragma omp simd` was promised to solve this issue but

#. It does not work.
#. It is not even defined for some compilers (OpenMP inside of cuda doesn't even
   make sense).

So LLAMA provides a macro called :cpp:`LLAMA_INDEPENDENT_DATA` which can be put
in front of loops to tell the underlying compiler that the data accesses in the
loop body are independent of each other -- and can savely be vectorized (what is
the goal in the end).

Datum domain iterating
----------------------

It is trivial to iterate over the user domain and although it is done at run
time the compiler can optimize a lot e.g. with tree vectorization or loop
unrolling, especially with the beforementioned macros.

It is also possible to iterate over the datum domain, even without some macro
hacks as shown before, totalling staying in our precious C++11 world. But this
can at the moment only be archieved with functional meta programming techniques,
making the code complicated and bloated. Even some simple iterating has to be
done recursively.

LLAMA provides a function to easy the pain (a bit) called :cpp:`llama::forEach`.
It takes a datum domain as compile time parameter and a functor as compile and
run time parameters and calls this functor for each leaf of the datum domain
tree, e.g.

.. code-block:: C++

    using DatumDomain = llama::DS <
        llama::DE < x, float >,
        llama::DE < y, float >,
        llama::DE < z, llama::DS <
            llama::DE <  low, short int >,
            llama::DE < high, short int >
        > >
    >;

    MyFunctor functor;

    // "functor" will be called for
    // * x
    // * y
    // * z.low
    // * z.high
    llama::forEach<DatumDomain>(functor);

Optionally a branch of the DatumDomain can be chosen to execute the functor on.
This is working both for addressing with names and `DatumCoord`.

.. code-block:: C++

    // "functor" will be called for
    // * z.low
    // * z.high
    llama::forEach< DatumDomain, z >(functor);

    // "functor" will be called for
    // * z.low
    llama::forEach< DatumDomain, z, low >(functor);

    // "functor" will be called for
    // * z.high
    llama::forEach< DatumDomain, llama::DatumCoord< 2, 1 > >(functor);

The functor type itself is a struct which provides the :cpp:`operator()` for
two different template parameters. The (run time) datum to work on and other
properties can be given as struct members. The template parameters are outer and
inner coordinates in the datum domain tree. The outer coordinate is what can be
given as template parameter(s) to :cpp:`llama::ForEach` after the datum domain
itself. However even if given as naming, the functor always gets a
:cpp:`DatumCoord`. The inner coord is the leaf coordinate based on the outer
coord. To get the needed global coodinate in the tree :cpp:`llama::DatumCoord`
provides a method called :cpp:`Cat` as seen in the next example functor.

.. code-block:: C++

    template<
        typename T_VirtualDatum,
        typename T_Value
    >
    struct SetValueFunctor
    {
        template<
            typename T_OuterCoord,
            typename T_InnerCoord
        >
        auto
        operator()(
            T_OuterCoord,
            T_InnerCoord
        )
        -> void
        {
            // the global coordinate in the tree is provided with "Cat"
            vd( typename T_OuterCoord::template Cat< T_InnerCoord >() ) = value;
        }
        T_VirtualDatum vd;
        T_Value const value;
    };

    // ...

    auto vd = view( 23, 43 );

    SetValueFunctor<
        decltype( vd ),
        float
    > functor( 1337.0f );

    llama::forEach<DatumDomain>(functor);

A more detailed example can be found in the
`simpletest example <https://github.com/ComputationalRadiationPhysics/llama/blob/master/examples/simpletest/simpletest.cpp>`_.

Copy
----

Especially when working with hardware accelerators such as GPUs or offloading
many core procressors, explicit copy operation calls for as big as possible
memory chunks are very important to reach best performance.

It is trivial to copy a view from one memory region to another if mapping and
size are identical. However if the mapping differs, in most of the
cases only elementwise copy operations will be possible as the memory patterns
are probably not compatible. There is a small class of remaining use cases where
the mapping is the same, but the size of the view is different or mappings are
very related to each other (e.g. both using struct of array, but one time with,
one time without padding). In those cases an optimized copy operation would be
possible in *theory*. However *practically* it is impossible to figure out the
biggest possible memory chunks to copy for LLAMA at compile time as the mappings
can always depend on run time parameters. E.g. a mapping could implement struct
of array if the view is bigger than :math:`255` elements, but use array of
struct for a smaller amount.

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

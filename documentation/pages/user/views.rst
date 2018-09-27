.. include:: ../common.rst

.. _label-view:

View
====

The view is the main data structure a LLAMA user will work with. It takes
coordinates in the user and datum domain and returns a reference to a datum
in memory which can be read or altered. For more easy use furthermore some
useful operations such as :cpp:`+=` are overloaded to operate on all datum
elements inside the datum domain at once.

.. _label-factory:

Factory
-------

The factory creates the view. For this it takes the domains, a
:ref:`mapping <label-mappings>` and an :ref:`allocator <label-allocators>`.

.. code-block:: C++

    using Mapping = ...; // see next section about mappings
    Mapping mapping( userDomainSize ); // see section about domains
    using Factory = llama::Factory<
        Mapping,
        llama::allocator::SharedPtr // see over next section about allocators
    >;
    auto view = Factory::allocView( mapping );

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

A direct call of the :cpp:`operator()` like
:cpp:`view( 1, 2, 3 )< color, g >()` is not possible unfortunately and would
look like this: :cpp:`view( 1, 2, 3 ).operator()< color, g >()` instead. So as
an explicit call is needed anyway LLAMA got an own function for this task.

Different algorithms have different requirements for accessing data, e.g. it
is also possible to access the user domain with one packed parameter like this

.. code-block:: C++


    view( { 1, 2, 3 } )( color(), g() ) = 1.0;
    // or
    UserDomain const pos{ 1, 2, 3 };
    view( pos )( color(), g() ) = 1.0;

If the naming in the datum domain is not important, may change (e.g. with the
same algorithm working in the RGB or CYK colour space) or is not available at
all (e.g. for a :cpp:`DatumArray`) or if the algorithm wants to iterate over the
datum domain (at compile time of course), also an adressing with the coordinate
inside the tree is possible like this:

.. code-block:: C++

    view( 1, 2, 3 )( llama::DatumCoord< 0, 1 >() ) = 1.0; // color.g
    // or
    view( 1, 2, 3 ).access< 0, 1 >() = 1.0; // color.g

Here the version with the explicit :cpp:`access` function call is even shorter.

VirtualDatum
^^^^^^^^^^^^

It may have attracted attention that the :cpp:`operator()` is overloaded twice
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
) and to increase this feeling some very useful operators are overloaded, too:

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

The most needed inplace operators ( :cpp:`+=`, :cpp:`-=`, :cpp:`*=`, :cpp:`/=`,
:cpp:`%=` ) are overloaded. Only inplace, because it is not trivial to create a
needed intermediate state out of a virtual datum (without expression templates).
These operators work between two virtual datums, even if they have different
datum domains. Every namings existing in both datum domains will be matched and
operated on. Every not matching pair is ignored, e.g.

.. code-block:: C++

    DD1 = llama::DS <
        llama::DS < llama::DE < pos
            llama::DE < x, float >
        > >,
        llama::DS < llama::DE < vel
            llama::DE < x, double >
        > >,
        llama::DE < x, int >
    >;

    DD2 = llama::DS <
        llama::DS < llama::DE < pos
            llama::DE < x, double >
        > >,
        llama::DS < llama::DE < mom
            llama::DE < x, double >
        > >
    >;

    // Let datum1 be using DD1 and datum2 using DD2.

    datum1 += datum2;
    // datum2.pos.x and only datum2.pos.x will be added to datum1.pos.x because
    // of pos.x existing in both datum domains although having different types.

The same operators are also overloaded for any other type so that
:cpp:`datum1 *= 7.0` will multiply 7 to every element in the datum domain.
Of course this may throw warnings about narrowing conversion. It is task of the
user to only use this if compatible.

A partly addressing of a virtual datum like :cpp:`datum1( color() ) *= 7.0`
would be handy, too, which is planned but not implemented yet.

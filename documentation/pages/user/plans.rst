.. include:: ../common.rst

Future plans & to do
====================

It was probably noticable that some functionalities of LLAMA are not feature
complete yet. The interfaces are basically fixed and hopefully slim enough to
make the LLAMA parts as independent as possible. They are not plans to change
the user interface as well.

So LLAMA can be used freely without the risk of big, design breaking changes in
the near future. However there is still a long way to go.

View and virtual datum interface
--------------------------------

One of the most relevent plans for end users is the extension of the view and
virtual datum functionalities. At the moment only five operators are overload,
in the future all arightmetic and logic operators shall be implemented -- at
least of they are inplace. :cpp:`view1( i ) = view2( j ) + view( k )` would
need intermediate temporary objects (e.g. on the stack) which may be bad or
expression templates. Both is *not* planned atm but not completely excluded as
well.

Explictly planned is to be able to work on intermediate objects between a full
view and a concrete element in the datum domain so that such expression are
valid:

.. code-block:: C++

    using UserDomain = llama::UserDomain< 2 >;
    using DatumDomain = llama::DatumStruct <
        llama::DatumElement < color, llama::DatumStruct <
            llama::DatumElement < r, float >,
            llama::DatumElement < g, float >,
            llama::DatumElement < b, float >
        > >,
        llama::DatumElement < alpha, char >
    >;

    // binding the first user domain dimension
    auto subview = view( 13 );
    auto vd1 = subview( 37 ); // same as view( 13, 37 );
    auto vd2 = view( 23, 42 );

    // working only on the color sub datum domain
    vd1( color() ) += vd2( color() );

    // binding the first datum domain hierarchie level
    auto subVd1 = vd1( color() );
    subVd1 = 0; //only resetting the color, but not the alpha value

Another thought is to be able to easily define sub datum domains out of existing
ones, e.g. taking only one branch of the type tree or removing branches.

Mappings
--------

There are still some common memory access patterns not yet implemented in
LLAMA like

* padding
* blocking
* striding

which will be implemented at least for the tree mapping interface. Furthermore
the struct of array mappings will get more detailed option to steer until which
tree level the run time annotation should be moved.

If it make sense furthermore an alternative mapping language interface could be
implemented if the tree mapping interface shows to not work out well.

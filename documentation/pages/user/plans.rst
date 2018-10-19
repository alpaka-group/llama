.. include:: ../common.rst

Future plans & to do
====================

It was probably noticable that some functionalities of LLAMA are not feature
complete yet. The interfaces are basically fixed and hopefully slim enough to
make the LLAMA parts as independent as possible. They are no plans to change
the user interface of the views and virtual view as well (except extending).

So LLAMA can be used freely without the risk of big, design breaking changes in
the near future. However there is still a long way to go.

View and virtual datum interface
--------------------------------

One of the most relevent plans for end users is the extension of the view and
virtual datum functionalities. At the moment only six inplace and six logical
operators are overload. In the future all arightmetic and logic operators shall
be implemented -- at least of they are inplace.
:cpp:`view1( i ) = view2( j ) + view( k )` would need intermediate temporary
objects (e.g. on the stack) which may be bad or expression templates. Both is
not **planned** atm but **not completely excluded** as well.

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

If it makes sense furthermore an alternative mapping language interface could be
implemented if the tree mapping interface shows to not work out well.

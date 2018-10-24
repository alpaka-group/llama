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

Right now only the inplace operations like :cpp:`+=`, the logical operations and
the five most important binary operations are supported. In the future all
binary (and unary?) operators shall be overloaded.

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

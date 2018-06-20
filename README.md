LLAMA – Low Level Abstraction of Memory Access
==============================================

Motivation
----------

We face the problem that different architectures these days perform best with
different memory access patterns, but as projects may last for decades while
new architectures rises and fall, it is dangerous to settle for one kind of
access. It is well-known that accessing complex data in a struct of array (SoA)
manner is most of the times faster than as array of structs (AoS):
```C++
// Array of Struct   |   // Struct of Array
struct               |   struct
{                    |   {
    float r, g, b;   |       float r[64][64], g[64][64], b[64][64];
    char a;          |       char a[64][64];
} image[64][64];     |   } image;
```

Even this very easy decision between SoA and AoS has quite different access
patterns, just compare `image[x][y].r` with `image.r[x][y]`. However for this
problem research and ready to use libraries already exist like
[SoAx](https://www.sciencedirect.com/science/article/pii/S0010465517303983)

However there are more useful mappings than SoA and AoS such as blocking of
memory (like partly using SoA inside an AoS approach), strided access of data
(e.g. odd indexes after each other), padding and more.

Often software is using a random mix of different heterogenous memory regions of
CPU, GPU, caches or network cards. A data layout optimized for a specific CPU
may be inefficient on a GPU or only slow to transfer over network. So a mixed
layout, not optimal for each part but the fastest trade-off, may make sense.
Again: This layout is highly dependent on the architecture, the scaling of the
problem and of course the chosen algorithm – and most probably not trivially
guessable.

Furthermore other third party libraries may expect specific memory layouts as
interface, which will most probably differ from library to library.

Challenges
----------

This results in these challenges LLAMA tries to address:

* splitting of algorithmic view of data and the actual mapping in the background
  so that different layouts may be chosen **without touching the algorithm at
  all**.
* as it is well-known from C and C++ and because of this often the way
  programmers think of data, LLAMA shall *look* like AoS although the mapping
  will be different quite surely.
* to be compatible with as most architectures, softwares, compilers and third
  party libraries as possible, LLAMA is only using valid C++11 syntax. The
  whole description of the layout and the mapping is down with C++11 template
  programming (in contrast e.g. to fancy macro magic which is slow and hard to
  maintain).
* LLAMA shall be extensible in the sense of working togehter with new software
  but also memory needed for new architectures.
* as it is the most easy way to write architecture independet but performant
  code, LLAMA should work well with auto vectorization approaches.

Concept
-------

To archieve these challenges LLAMA has a strict splitting of the "array" and
the "struct" part and to not mistakes these with the native C/C++ arrays and
structs, LLAMA defines

* the **user domain (UD)** as n-dimensional "array" with a **compile time
  dimension** but **run time size** per dimension and
* the **datum domain (DD)** as "struct" which is defined as a **compile time
  tree** (seen later).

To know how to map these both independent domains to memory, **mappings** are
defined. These mappings map a (run time) address in UD and a (compile time)
address in DD to byte addresses of potentially different memory regions
`UD` ⨯ `DD` → `memory region` ⨯ `byte address`. The mapping also defines how
much memory is needed, which can then be used to allocate it. The **allocator**
object is an important connection to other libraries such as
[Alpaka](https://github.com/ComputationalRadiationPhysics/alpaka) as LLAMA
itself has no knowledge about memory regions.

The **factory** takes the above-mentioned objects and creates a **view**, which
is a user accessable container of memory with the given attributes. It can be
used quite similar to C++ containers or C arrays of structs.

![The factory creates a view out of the user domain, datum domain, mapping
and the allocator](./documentation/images/factory.svg)

A definition of a datum domain looks like this
```C++
struct color {};
struct alpha {};
struct r {};
struct g {};
struct b {};

using namespace llama;

using Pixel = DatumStruct <
    DatumElement < color, DatumStruct <
        DatumElement < r, float >,
        DatumElement < g, float >,
        DatumElement < b, float >,
    > >,
    DatumElement < alpha, char >
>;
```

In pure C/C++ this may look like this
```C++
struct Pixel {
    struct {
        float r,g,b;
    } color;
    char alpha;
};
```

However it is not possible to iterate over struct members in C++11, so we need
to define the DD in the above-shown way. The naming of the members needs to be
predefined to detect the semantically same member in differend DDs. Furthermore
these namings can be encapsulated inside namespaces.

The DD tree would look like this:

![Datum domain as tree](./documentation/images/layout_tree.svg)

The user domain is defined more easy like this
```C++
using UserDomain = llama::UserDomain< 2 >;
const UserDomain userDomain{ 64, 64 };
```

For an address in the user domain the view can return a **virtal datum** which
feels like a element in an n-dimensiona array but may still be distributed in
memory:
```C++
auto datum = view( { 23, 42 } );
```

With the compile time address in the datum domain, a definite memory access can
be done. All these access methods write the same value:
```C++
datum( color(), g() ) = 1.0f; //access with hierarichal dd namings
datum( llama::DatumCoord< 0, 1 >() ) = 1.0f; //access with dd tree coordinate
datum.access( color(), g() ) = 1.0f; //access with explicit access function
datum.access< color, g >() = 1.0f; //access with explicit template namings
datum.access< llama::DatumCoord< 0, 1 > >() = 1.0f; //same with tree coordinate
```

Furthermore virtual data can directly be changed without a adress in the DD
like this
```C++
datum *= 5.0f;
datum1 += datum2;
```
where `datum1` and `datum2` don't need to have the same datum domain at all!
For every element in `datum1` also found in `datum2` (at compile time) the
`+=` operation is executed. Of the DD are without overlap, nothing happens at
all.

Mapping description
-------------------
TODO

Tree mappings
-------------
TODO

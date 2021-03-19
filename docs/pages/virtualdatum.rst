.. include:: common.rst

.. _label-view:

VirtualDatum
============

During a view accesses like :cpp:`view(1, 2, 3)(color{}, g{})` an intermediate object is needed for this to work.
This object is a :cpp:`llama::VirtualDatum`.

.. code-block:: C++

    using Pixel = llama::DS<
        llama::DE<color, llama::DS<
            llama::DE<r, float>,
            llama::DE<g, float>,
            llama::DE<b, float>
        >>,
        llama::DE<alpha, char>
    >;
    // ...

    auto vd = view(1, 2, 3);

    vd(color{}, g{}) = 1.0;
    // or:
    auto vdColor = vd(color{});
    float& g = vdColor(g{});
    g = 1.0;

Supplying the array domain coordinates to a view access returns such a :cpp:`llama::VirtualDatum`, storing this array domain coordiante.
This object can be thought of like a datum in the :math:`N`-dimensional array domain space,
but as the elements of this datum may not be contiguous in memory, it is not a real object in the C++ sense and thus called virtual.

Accessing subparts of a :cpp:`llama::VirtualDatum` is done using `operator()` and the tag types from the datum domain.

If an access describes a final/leaf element in the datum domain, a reference to a value of the corresponding type is returned.
Such an access is called terminal. If the access is non-termian, i.e. it does not yet reach a leaf in the datum domain tree,
another :cpp:`llama::VirtualDatum` is returned, binding the tags already used for navigating down the datum domain.

A :cpp:`llama::VirtualDatum` can be used like a real local object in many places. It can be used as a local variable, copied around, passed as an argument to a function (as seen in the
`nbody example <https://github.com/alpaka-group/llama/blob/master/examples/nbody/nbody.cpp>`_), etc.
In general, :cpp:`llama::VirtualDatum` is a value type that represents a reference, similar to an iterator in C++ (:cpp:`llama::One` is a notable exception).


One
---

:cpp:`llama::One<DatumDomain>` is a shortcut to create a scalar :cpp:`llama::VirtualDatum`.
This is useful when we want to have a single datum instance e.g. as a local variable.

    llama::One<Pixel> datum;
    datum(color{}, g{}) = 1.0;
    auto datum2 = datum; // independent copy

Technically, :cpp:`llama::One` is a :cpp:`llama::VirtualDatum` which stores a scalar :cpp:`llama::View` inside, using the mapping :cpp:`llama::mapping::One`.
This also has the unfortunate consequence that a :cpp:`llama::One` is now a value type with deep-copy semantic.
We might address this inconsistency at some point.


Arithmetic and logical operatores
---------------------------------

:cpp:`llama::VirtualDatum` overloads several arithmetic and logical operatores:

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

    datum1(vel{}) *= datum2(mom{});
    // datum2.mom.x will be multiplied to datum2.vel.x as the first part of the
    // datum domain coord is explicit given and the same afterwards

The discussed operators are also overloaded for types other than :cpp:`llama::VirtualDatum` as well so that
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
:cpp:`datum1(color{})` itself returns a new virtual datum with the first datum domain coordiante (:cpp:`color`) being bound.
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
`nbody example <https://github.com/alpaka-group/llama/blob/master/examples/nbody/nbody.cpp>`_
to update the particle velocity based on the distances of particles and to
update the position after one time step movement with the velocity.


Tuple interface
---------------

WARNING: This is an experimental feature and might completely change in the future.

A struct in C++ can be modelled by a :cpp:`std::tuple` with the same types as the struct's members.
A :cpp:`llama::VirtualDatum` behaves like a reference to a struct (i.e. the datum) which is decomposed into it's members.
We can therefore not form a single reference to such a datum, but references to the individual members.
Organizing these references inside a :cpp:`std::tuple` in the same way the datum is represented in the datum domain gives us an alternative to a :cpp:`llama::VirtualDatum`.
Mind that creating such a :cpp:`std::tuple` already invokes the mapping function, regardless of whether an actual memory access occurs through the constructed reference later.
However, such dead address computations are eliminated by most compilers during optimization.

.. code-block:: C++

    auto datum = view(1, 2, 3);
    std::tuple<std::tuple<float&, float&, float&>, char&> = datum.asTuple();
    std::tuple<float&, float&, float&, char&> = datum.asFlatTuple();
    auto [r, g, b, a] = datum.asFlatTuple();

Additionally, if the user already has types supporting the tuple interface, :cpp:`llama::VirtualDatum` can integreate with these using the :cpp:`load()`, :cpp:`loadAs<T>()` and :cpp:`store(T)` functions.

.. code-block:: C++

    struct MyPixel {
        struct {
            float r, g, b;
        } color;
        char alpha;
    };
    // implement std::tuple_size<MyPixel>, std::tuple_element<MyPixel> and get(MyPixel)

    auto datum = view(1, 2, 3);

    MyPixel p1 = datum.load(); // constructs MyPixel from 3 float& and 1 char&
    auto p2 = datum.loadAs<MyPixel>(); // same

    p1.alpha = 255;
    datum.store(p1); // tuple-element-wise assignment from p1 to datum.asFlatTuple()

Keep in mind that the load and store functionality always reads/writes all elements referred to by a :cpp:`llama::VirtualDatum`.


Structured bindings
-------------------

WARNING: This is an experimental feature and might completely change in the future.

A :cpp:`llama::VirtualDatum` implementes the tuple interface by providing corresponding specializations of :cpp:`std::tuple_size`, :cpp:`std::tuple_element` and a `llama::get<I>(llama::VirtualDatum)`free functions.
This allows a :cpp:`llama::VirtualDatum` to be destructured:

.. code-block:: C++

    auto datum = view(1, 2, 3);
    auto [color, a] = datum; // color is another VirtualDatum, a is a char&, 1 call to mapping function
    auto [r, g, b] = color; // r, g, b are float&, 3 calls to mapping function

Contrary to destructuring a tuple generated by calling :cpp:`asTuple()` or :cpp:`asFlatTuple()`,
the mapping function is not invoked for other instances of :cpp:`llama::VirtualDatum` created during the destructuring.
The mapping function is just invoked to form references for terminal accesses.

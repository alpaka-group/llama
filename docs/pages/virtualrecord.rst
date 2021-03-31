.. include:: common.rst

.. _label-virtualrecord:

VirtualRecord
============

During a view accesses like :cpp:`view(1, 2, 3)(color{}, g{})` an intermediate object is needed for this to work.
This object is a :cpp:`llama::VirtualRecord`.

.. code-block:: C++

    using Pixel = llama::Record<
        llama::Field<color, llama::Record<
            llama::Field<r, float>,
            llama::Field<g, float>,
            llama::Field<b, float>
        >>,
        llama::Field<alpha, char>
    >;
    // ...

    auto vd = view(1, 2, 3);

    vd(color{}, g{}) = 1.0;
    // or:
    auto vdColor = vd(color{});
    float& g = vdColor(g{});
    g = 1.0;

Supplying the array domain coordinates to a view access returns such a :cpp:`llama::VirtualRecord`, storing this array domain coordiante.
This object can be thought of like a record in the :math:`N`-dimensional array domain space,
but as the fields of this record may not be contiguous in memory, it is not a real object in the C++ sense and thus called virtual.

Accessing subparts of a :cpp:`llama::VirtualRecord` is done using `operator()` and the tag types from the record dimension.

If an access describes a final/leaf element in the record dimension, a reference to a value of the corresponding type is returned.
Such an access is called terminal. If the access is non-termian, i.e. it does not yet reach a leaf in the record dimension tree,
another :cpp:`llama::VirtualRecord` is returned, binding the tags already used for navigating down the record dimension.

A :cpp:`llama::VirtualRecord` can be used like a real local object in many places. It can be used as a local variable, copied around, passed as an argument to a function (as seen in the
`nbody example <https://github.com/alpaka-group/llama/blob/master/examples/nbody/nbody.cpp>`_), etc.
In general, :cpp:`llama::VirtualRecord` is a value type that represents a reference, similar to an iterator in C++ (:cpp:`llama::One` is a notable exception).


One
---

:cpp:`llama::One<RecordDim>` is a shortcut to create a scalar :cpp:`llama::VirtualRecord`.
This is useful when we want to have a single record instance e.g. as a local variable.

    llama::One<Pixel> pixel;
    pixel(color{}, g{}) = 1.0;
    auto pixel2 = pixel; // independent copy

Technically, :cpp:`llama::One` is a :cpp:`llama::VirtualRecord` which stores a scalar :cpp:`llama::View` inside, using the mapping :cpp:`llama::mapping::One`.
This also has the unfortunate consequence that a :cpp:`llama::One` is now a value type with deep-copy semantic.
We might address this inconsistency at some point.


Arithmetic and logical operatores
---------------------------------

:cpp:`llama::VirtualRecord` overloads several arithmetic and logical operatores:

.. code-block:: C++

    auto record1 = view(1, 2, 3);
    auto record2 = view(3, 2, 1);

    record1 += record2;
    record1 *= 7.0; //for every element in the record dimension

    foobar(record2);

    //With this somewhere else:
    template<typename VirtualRecord>
    void foobar(VirtualRecord vr)
    {
        vr = 42;
    }

The most common compount assignment operators ( :cpp:`=`, :cpp:`+=`, :cpp:`-=`, :cpp:`*=`,
:cpp:`/=`, :cpp:`%=` ) are overloaded. These operators directly write into the
corresponding view. Furthermore several arithmetic operators ( :cpp:`+`, :cpp:`-`,
:cpp:`*`, :cpp:`/`, :cpp:`%` ) are overloaded too, but they return a temporary object
on the stack. Althought this temporary value has a basic struct-mapping without padding and
probaly being not compatible to the mapping of the view at all, the compiler
will most probably be able to optimize the data accesses anyway as it has full
knowledge about the data in the stack and can cut out all temporary operations.

These operators work between two virtual records, even if they have
different record dimensions. It is even possible to work on parts of a virtual
record. This returns a virtual record with the first coordinate in the record dimension bound.
Every tag existing in both record dimensions will be matched and operated on.
Every non-matching tag is ignored, e.g.

.. code-block:: C++

    using RecordDim1 = llama::Record<
        llama::Record<llama::Field<pos
            llama::Field<x, float>
        >>,
        llama::Record<llama::Field<vel
            llama::Field <x, double>
        >>,
        llama::Field <x, int>
    >;

    using RecordDim2 = llama::Record<
        llama::Record<llama::Field<pos
            llama::Field<x, double>
        >>,
        llama::Record<llama::Field<mom
            llama::Field<x, double>
        >>
    >;

    // Let assume record1 using RecordDim1 and record2 using RecordDim2.

    record1 += record2;
    // record2.pos.x and only record2.pos.x will be added to record1.pos.x because
    // of pos.x existing in both record dimensions although having different types.

    record1(vel{}) *= record2(mom{});
    // record2.mom.x will be multiplied to record2.vel.x as the first part of the
    // record dimension coord is explicit given and the same afterwards

The discussed operators are also overloaded for types other than :cpp:`llama::VirtualRecord` as well so that
:cpp:`record1 *= 7.0` will multiply 7 to every element in the record dimension.
This feature should be used with caution!

The comparison operators :cpp:`==`, :cpp:`!=`, :cpp:`<`, :cpp:`<=`, :cpp:`>`
and :cpp:`>=` are overloaded too and return :cpp:`true` if
the operation is true for **all** pairs of fields with equal tag.
Let's examine this deeper in an example:

.. code-block:: C++

    using A = llama::Record <
        llama::Field < x, float >,
        llama::Field < y, float >
    >;

    using B = llama::Record<
        llama::Field<z, double>,
        llama::Field<x, double>
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

A partial addressing of a virtual record like :cpp:`record1(color{}) *= 7.0` is also possible.
:cpp:`record1(color{})` itself returns a new virtual record with the first record dimension coordiante (:cpp:`color`) being bound.
This enables e.g. to easily add a velocity to a position like this:

.. code-block:: C++

    using Particle = llama::Record<
        llama::Field<pos, llama::Record<
            llama::Field<x, float>,
            llama::Field<y, float>,
            llama::Field<z, float>
        >>,
        llama::Field<vel, llama::Record<
            llama::Field<x, double>,
            llama::Field<y, double>,
            llama::Field<z, double>
        >>,
    >;

    // Let record be a virtual record with the record dimension "Particle".

    record(pos{}) += record(vel{});

This is e.g. used in the
`nbody example <https://github.com/alpaka-group/llama/blob/master/examples/nbody/nbody.cpp>`_
to update the particle velocity based on the distances of particles and to
update the position after one time step movement with the velocity.


Tuple interface
---------------

WARNING: This is an experimental feature and might completely change in the future.

A struct in C++ can be modelled by a :cpp:`std::tuple` with the same types as the struct's members.
A :cpp:`llama::VirtualRecord` behaves like a reference to a struct (i.e. the record) which is decomposed into it's members.
We can therefore not form a single reference to such a record, but references to the individual members.
Organizing these references inside a :cpp:`std::tuple` in the same way the record is represented in the record dimension gives us an alternative to a :cpp:`llama::VirtualRecord`.
Mind that creating such a :cpp:`std::tuple` already invokes the mapping function, regardless of whether an actual memory access occurs through the constructed reference later.
However, such dead address computations are eliminated by most compilers during optimization.

.. code-block:: C++

    auto record = view(1, 2, 3);
    std::tuple<std::tuple<float&, float&, float&>, char&> = record.asTuple();
    std::tuple<float&, float&, float&, char&> = record.asFlatTuple();
    auto [r, g, b, a] = record.asFlatTuple();

Additionally, if the user already has types supporting the tuple interface, :cpp:`llama::VirtualRecord` can integreate with these using the :cpp:`load()`, :cpp:`loadAs<T>()` and :cpp:`store(T)` functions.

.. code-block:: C++

    struct MyPixel {
        struct {
            float r, g, b;
        } color;
        char alpha;
    };
    // implement std::tuple_size<MyPixel>, std::tuple_element<MyPixel> and get(MyPixel)

    auto record = view(1, 2, 3);

    MyPixel p1 = record.load(); // constructs MyPixel from 3 float& and 1 char&
    auto p2 = record.loadAs<MyPixel>(); // same

    p1.alpha = 255;
    record.store(p1); // tuple-element-wise assignment from p1 to record.asFlatTuple()

Keep in mind that the load and store functionality always reads/writes all elements referred to by a :cpp:`llama::VirtualRecord`.


Structured bindings
-------------------

WARNING: This is an experimental feature and might completely change in the future.

A :cpp:`llama::VirtualRecord` implementes the tuple interface by providing corresponding specializations of :cpp:`std::tuple_size`, :cpp:`std::tuple_element` and a `llama::get<I>(llama::VirtualRecord)`free functions.
This allows a :cpp:`llama::VirtualRecord` to be destructured:

.. code-block:: C++

    auto record = view(1, 2, 3);
    auto [color, a] = record; // color is another VirtualRecord, a is a char&, 1 call to mapping function
    auto [r, g, b] = color; // r, g, b are float&, 3 calls to mapping function

Contrary to destructuring a tuple generated by calling :cpp:`asTuple()` or :cpp:`asFlatTuple()`,
the mapping function is not invoked for other instances of :cpp:`llama::VirtualRecord` created during the destructuring.
The mapping function is just invoked to form references for terminal accesses.

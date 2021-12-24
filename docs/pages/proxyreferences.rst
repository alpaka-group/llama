.. include:: common.rst

.. _label-proxyreferences:

Proxy references
================

The story of :cpp:`std::vector<bool>`
----------------------------------------

When we want to refer to an object of type :cpp:`T` somewhere in memory,
we can form a reference to that object using the language built-in reference :cpp:`T&`.
This also holds true for containers, which often maintain larger portions of memory containing many objects of type :cpp:`T`.
Given an index, we can obtain a reference to one such :cpp:`T` living in memory:

.. code-block:: C++

    std::vector<T> obj(100);
    T& ref = obj[42];

The reference :cpp:`ref` of type :cpp:`T&` refers to an actual object of type :cpp:`T` which is truly manifested in memory.

Sometimes however, we choose to store the value of a :cpp:`T` in a different way in memory, not as an object of type :cpp:`T`.
The most prominent example of such a case is :cpp:`std::vector<bool>`, which uses bitfields to store the values of the booleans,
thus decreasing the memory required for the data structure.
However, since :cpp:`std::vector<bool>` does not store objects of type :cpp:`bool` in memory,
we can now longer form a :cpp:`bool&` to one of the vectors elements:

.. code-block:: C++

    std::vector<bool> obj(100);
    bool& ref = obj[42]; // compile error

The proposed solution in this case is to replace the :cpp:`bool&` by an object representing a reference to a :cpp:`bool`.
Such an object is called a proxy reference.
Because some standard containers may use proxy references for some contained types, when we write generic code,
it is advisable to use the corresponding :cpp:`reference` alias provided by them, or to use a forwarding reference:

.. code-block:: C++

    std::vector<T> obj(100);
    std::vector<T>::reference ref1 = obj[42]; // works for any T including bool
    auto&&                    ref2 = obj[42]; // binds to T& for real references,
                                              // or proxy references returned by value

Although :cpp:`std::vector<bool>` is notrious for this behavior of its references,
more such data structures exist (e.g. :cpp:`std::bitset`) or started to appear in recent C++ standards and its proposals.
E.g. in the area of `text encodings <https://thephd.dev/proxies-references-gsoc-2019>`_,
or `the zip range adaptors <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p2214r0.html#a-tuple-that-is-writable>`_.


Working with proxy references
-----------------------------

A proxy reference is usually a value-type with reference semantic.
Thus, a proxy reference can be freely created, copied, moved and destroyed.
Their sole purpose is to give access to a value they refer to.
They usually encapsulate a reference to some storage and computations to be performed when writing or reading through the proxy reference.
Write access to a refered value of type :cpp:`T` is typically given via an assignment operator from :cpp:`T`.
Read access is given by a (non-explicit) conversion operator to :cpp:`T`.

.. code-block:: C++

    std::vector<bool> v(100);
    auto&& ref = v[42];

    ref = true;    // write: invokes std::vector<bool>::reference::operator=(bool)
    bool b1 = ref; // read:  invokes std::vector<bool>::reference::operator bool()

    auto  ref2 = ref;  // takes a copy of the proxy reference (!!!)
    auto& ref3 = ref2; // references (via the language build-in l-value reference) the proxy reference ref2

    for (auto&& element : v)
       bool b = element;

Mind, that we explicitely state :cpp:`bool` as the type of the resulting value.
If we use :cpp:`auto` instead, we would take a copy of the reference object, not the value.

Proxy references in LLAMA
-------------------------

By handing out references to access contained objects, LLAMA views are similar to standard C++ containers.
For references to whole records, LLAMA views hand out virtual records.
Although a virtual record models a reference to a struct (= record) in memory, this struct is not physically manifested in memory.
This allows mappings the freedom to arbitrarily arrange how the data for a struct is stored.
A virtual record in LLAMA is thus conceptually a proxy reference.
An exception is however made for read/write access in the current API, which is governed by the :cpp:`load()` and :cpp:`store()` member functions.
We might change this in the future.

.. code-block:: C++

    auto view = llama::allocView(...);
    auto vr = view(1, 2, 3); // vr is a VirtualRecord, a proxy reference
    Pixel p = vr.load(); // read access
    vr.store(p);         // write access

Similarly, some mappings choose a different in-memory representation for the field types in the leaves of the record dimension.
Examples are the :cpp:`Bytesplit`, :cpp:`ChangeType`, :cpp:`BitPackedIntSoa` or  :cpp:`BitPackedFloatSoa` mappings.
These mappings even return a proxy reference for terminal accesses:

.. code-block:: C++

    auto&& ref = vr(color{}, r{}); // may be a float& or a proxy reference object, depending on the mapping

Thus, when you want to write truly generic code with LLAMA's views, please keep these guidelines in mind:

 * Each non-terminal access on a view returns a virtual record, which is a value-type with reference semantic.
 * Each terminal access on a view may return an l-value reference or a proxy reference.
   Thus use :cpp:`auto&&` to handle both cases.
 * Explicitely specify the type of copies of individual fields you want to make from references obtains from a LLAMA view.
   This avoids accidentially coping a proxy reference.

Concept
-------

Proxy references in LLAMA fullfill the following concept:

.. code-block:: C++

    template <typename R>
    concept ProxyReference = requires(R r) {
        typename R::value_type;
        { static_cast<typename R::value_type>(r) } -> std::same_as<typename R::value_type>;
        { r = typename R::value_type{} } -> std::same_as<R&>;
    };

That is, the provide a member type :cpp:`value_type`,
which indicates the type of the values which can be loaded and stored through the proxy reference.
Furthmore, a proxy reference can be converted to its value type (thus calling :cpp:`operator value_type ()`)
or assigned an instance of its value type.

Arithmetic on proxy references and ProxyRefOpMixin
--------------------------------------------------

An additional feature of normal references in C++ is that they can be used as operands for certain operators:

.. code-block:: C++

    auto&& ref = ...;
    T = ref + T(42); // works for normal and proxy references
    ref++;           // normally, works only for normal references
    ref *= 2;        // -||-
                     // both work in LLAMA due to llama::ProxyRefOpMixin

Proxy references cannot be used in compound assignment and increment/decrement operators unless they provide overloads for these operators.
To cover this case, LLAMA provides the `CRTP <https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern>`_ mixin :cpp:`llama::ProxyRefOpMixin`,
which a proxy reference type can inherit from, to supply the necessary operators.
All proxy reference types in LLAMA inherit from :cpp:`llama::ProxyRefOpMixin` to supply the necessary operators.
If you define your own computed mappings returning proxy references,
make sure to inherit your proxy reference types from :cpp:`llama::ProxyRefOpMixin`.

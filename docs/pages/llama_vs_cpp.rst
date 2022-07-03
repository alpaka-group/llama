.. include:: common.rst

LLAMA vs. C++
=============

LLAMA tries hard to provide experience and constructs similar to native C++.
The following tables compare how various constructs in C++ translate to LLAMA:

Containers and views
--------------------

.. list-table::
    :header-rows: 1
    :class: tight-table

    * - Construct
      - Native C++
      - LLAMA
      - LLAMA (alternative)
    * - Defining structs/records
      - .. code:: C++

          struct VecCpp {
            float x;
            float y;
          };
          struct ParticleCpp {
            VecCpp pos;
            float mass;
            bool flags[3];
          };
      - .. code:: C++

          struct X{}; struct Y{}; struct Pos{}; struct Mass{}; struct Flags{};
          using VecRec = llama::Record<
            llama::Field<X, float>,
            llama::Field<Y, float>
          >;
          using ParticleRec = llama::Record<
            llama::Field<Pos, VecRec>,
            llama::Field<Mass, float>,
            llama::Field<Flags, bool[3]>
          >;
      -
    * - Defining array extents
      - .. code:: C++

          using size_type = ...;
          size_type n = ...;
      - .. code:: C++

          using ArrayExtents = ...;
          ArrayExtents n = ...;
      -
    * - Defining the memory layout
      - \-
      - .. code:: C++

          using Mapping = ...;
          Mapping m(n, ...);
      -
    * - A collection of n things in memory
      - .. code:: C++

          std::vector<ParticleCpp> view(n);
      - .. code:: C++

          auto view = llama::allocView(m);
      - .. code:: C++

          llama::View<ArrayExtents, ParticleRec, ...> view;

        Useful for static array dimensions.

Values and references
---------------------

.. list-table::
    :header-rows: 1
    :class: tight-table

    * - Construct
      - Native C++
      - LLAMA
      - LLAMA (alternative)
      - wrong
    * - Declare single local record
      - .. code:: C++

          ParticleCpp p;
      - .. code:: C++

          llama::One<ParticleRec> p
      - .. code:: C++

          ParticleCpp p;

        Or any type layout compatible type supporting the tuple interface.
      - .. code:: C++

          ParticleRec p;

        ParticleRec is an empty struct (a type list)!
    * - Copy memory -> local
      - .. code:: C++

          p = view[i];
      - .. code:: C++

          p = view[i];
      - .. code:: C++

          p = view[i];

        Assigns field by field using tuple interface.
      -
    * - Copy local -> memory
      - .. code:: C++

          view[i] = p;
      - .. code:: C++

          view[i] = p;
      - .. code:: C++

          view[i] = p;

        Assigns field by field using tuple interface.
      -
    * - Copy a single record from memory to local
      - .. code:: C++

          ParticleCpp p = view[i];

      - .. code:: C++

          llama::One<ParticleRec> p = view[i];

      - .. code:: C++

          ParticleCpp p = view[i];

        Assigns field by field using tuple interface
      - .. code:: C++

          auto p = view[i];

        :cpp:`p` is a reference, not a copy!
    * - Create a reference to a single record in memory
      - .. code:: C++

          ParticleCpp& p = view[i];
      - .. code:: C++

          auto p = view[i];
          // decltype(p) == llama::RecordRef<...>
      - .. code:: C++

          auto&& p = view[i];
      - .. code:: C++

          auto& p = view[i];

        Compilation error!

    * - Copy a single sub-record from memory to local
      - .. code:: C++

          VecCpp v = view[i].pos;
      - .. code:: C++

          llama::One<VecRec> v = view[i](Pos{});
      - .. code:: C++

          VecRec v = view[i](Pos{});

        Assigns field by field using tuple interface.
      - .. code:: C++

          auto v = view[i](Pos{});

        :cpp:`v` is a reference, not a copy!
    * - Create a reference to a single sub-record in memory
      - .. code:: C++

          VecCpp& v = view[i].pos;
      - .. code:: C++

          auto v = view[i](Pos{});
          // decltype(v) == llama::RecordRef<...>
      - .. code:: C++

          auto&& v = view[i](Pos{});
      - .. code:: C++

          auto& p = view[i](Pos{});

        Compilation error!

    * - Copy a single record leaf field from memory to local
      - .. code:: C++

          float y = view[i].pos.y;
      - .. code:: C++

          float y = view[i](Pos{}, Y{});
      - .. code:: C++

          float y = view[i](Pos{})(Y{});
      -
    * - Create a reference to a single leaf field in memory
      - .. code:: C++

          float& y = view[i].pos.y;
      - .. code:: C++

          float& y = view[i](Pos{});
      - .. code:: C++

          auto&& y = view[i](Pos{});
      - .. code:: C++

          auto y = view[i](Pos{});

        :cpp:`y` is a copy (of type :cpp:`float`), not a reference!
    * - Create a copy of a single local record
      - .. code:: C++

          auto p2 = p;
      - .. code:: C++

          auto p2 = p;
      -
      -
    * - Create a reference to a single local record
      - .. code:: C++

          auto& r = p;
      - .. code:: C++

          auto r = p();
        Access with an empty tag list.
      -
      -


Notice that the use of :cpp:`auto` to declare a local copy of a value read through a reference, e.g. :cpp:`auto pos = view[i].pos; // copy`, does not work as expected in LLAMA.
LLAMA makes extensive use of proxy reference types (including :cpp:`llama::RecordRef`),
where a reference is sometimes represented as a value and sometimes as a real C++ reference.
The only consistent way to deal with this duality in LLAMA is the use a forwarding reference :cpp:`auto&&`
when we want to have a reference (native or proxy) into a LLAMA data structure,
and to use a concrete type when we want to make a copy.

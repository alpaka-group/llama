#pragma once

#include "Core.hpp"
#include "RecordRef.hpp"
#include "macros.hpp"
#include "mapping/SoA.hpp"

#include <type_traits>

namespace llama
{
    /// Traits of a specific Simd implementation. Please specialize this template for the SIMD types you are going to
    /// use in your program.
    /// Each specialization SimdTraits<Simd> must provide:
    /// * an alias `value_type` to indicate the element type of the Simd.
    /// * a `static constexpr size_t lanes` variable holding the number of SIMD lanes of the Simd.
    /// * a `static auto loadUnalinged(const value_type* mem) -> Simd` function, loading a Simd from the given memory
    /// address.
    /// * a `static void storeUnaligned(Simd simd, value_type* mem)` function, storing the given Simd to a given
    /// memory address.
    template<typename Simd, typename SFINAE = void>
    struct SimdTraits
    {
        static_assert(sizeof(Simd) == 0, "Please specialize SimdTraits for the type Simd");
    };

    template<typename T>
    struct SimdTraits<T, std::enable_if_t<std::is_arithmetic_v<T>>>
    {
        using value_type = T;

        inline static constexpr std::size_t lanes = 1;

        static LLAMA_FORCE_INLINE auto loadUnaligned(const T* mem) -> T
        {
            return *mem;
        }

        static LLAMA_FORCE_INLINE void storeUnaligned(T t, T* mem)
        {
            *mem = t;
        }
    };

    /// The number of SIMD simdLanes the given SIMD vector or \ref Simd<T> has. If Simd is not a structural \ref Simd
    /// or \ref SimdN, this is a shortcut for SimdTraits<Simd>::lanes.
    template<typename Simd, typename SFINAE = void>
    inline constexpr auto simdLanes = SimdTraits<Simd>::lanes;

    namespace internal
    {
        template<std::size_t N, template<typename, /* std::integral */ auto> typename MakeSizedSimd>
        struct BindMakeSizedSimd
        {
            template<typename U>
            using fn = MakeSizedSimd<U, N>;
        };

        template<
            typename RecordDim,
            std::size_t N,
            template<typename, /* std::integral */ auto>
            typename MakeSizedSimd>
        struct SimdizeNImpl
        {
            using type = TransformLeaves<RecordDim, internal::BindMakeSizedSimd<N, MakeSizedSimd>::template fn>;
        };

        template<typename RecordDim, template<typename, /* std::integral */ auto> typename MakeSizedSimd>
        struct SimdizeNImpl<RecordDim, 1, MakeSizedSimd>
        {
            using type = RecordDim;
        };
    } // namespace internal

    /// Transforms the given record dimension into a SIMD version of it. Each leaf field type will be replaced by a
    /// sized SIMD vector with length N, as determined by MakeSizedSimd. If N is 1, SimdizeN<T, 1, ...> is an alias for
    /// T.
    template<typename RecordDim, std::size_t N, template<typename, /* std::integral */ auto> typename MakeSizedSimd>
    using SimdizeN = typename internal::SimdizeNImpl<RecordDim, N, MakeSizedSimd>::type;

    /// Transforms the given record dimension into a SIMD version of it. Each leaf field type will be replaced by a
    /// SIMD vector, as determined by MakeSimd.
    template<typename RecordDim, template<typename> typename MakeSimd>
    using Simdize = TransformLeaves<RecordDim, MakeSimd>;

    /// Determines the number of simd lanes suitable to process all types occurring in the given record dimension. The
    /// algorithm ensures that even SIMD vectors for the smallest field type are filled completely and may thus require
    /// multiple SIMD vectors for some field types.
    /// @param RecordDim The record dimension to simdize
    /// @param MakeLargestSimd Type function creating the largest suitable SIMD type given a field type from the record
    /// dimension.
    template<typename RecordDim, template<typename> typename MakeLargestSimd>
    inline constexpr std::size_t simdLanesFor = []
    {
        std::size_t maxLanes = 0;
        boost::mp11::mp_for_each<boost::mp11::mp_transform<std::add_pointer_t, FlatRecordDim<RecordDim>>>(
            [&](auto* t)
            {
                using T = std::remove_reference_t<decltype(*t)>;
                maxLanes = std::max(maxLanes, simdLanes<MakeLargestSimd<T>>);
            });
        assert(maxLanes > 0);
        return maxLanes;
    }();

    /// Creates a SIMD version of the given type. Of T is a record dimension, creates a \ref One where each field is a
    /// SIMD type of the original field type. The SIMD vectors have length N. If N is 1, an ordinary \ref One of the
    /// record dimension T is created. If T is not a record dimension, a SIMD vector with value T and length N is
    /// created. If N is 1 (and T is not a record dimension), then T is produced.
    template<typename T, std::size_t N, template<typename, /* std::integral */ auto> typename MakeSizedSimd>
    using SimdN = typename std::conditional_t<
        isRecord<T> || internal::IsBoundedArray<T>::value,
        std::conditional_t<
            N == 1,
            boost::mp11::mp_identity<One<T>>,
            boost::mp11::mp_identity<One<SimdizeN<T, N, MakeSizedSimd>>>>,
        std::conditional_t<
            N == 1,
            boost::mp11::mp_identity<T>,
            boost::mp11::mp_identity<SimdizeN<T, N, MakeSizedSimd>>>>::type;

    /// Creates a SIMD version of the given type. Of T is a record dimension, creates a \ref One where each field is a
    /// SIMD type of the original field type.
    template<typename T, template<typename> typename MakeSimd>
    using Simd = typename std::conditional_t<
        isRecord<T> || internal::IsBoundedArray<T>::value,
        boost::mp11::mp_identity<One<Simdize<T, MakeSimd>>>,
        boost::mp11::mp_identity<Simdize<T, MakeSimd>>>::type;

    namespace internal
    {
        template<std::size_t S>
        struct SizeEqualTo
        {
            template<typename Simd>
            using fn = std::bool_constant<simdLanes<Simd> == S>;
        };
    } // namespace internal

    /// Specialization for Simd<RecordDim>. Only works if all SIMD types in the fields of the record dimension have the
    /// same size.
    template<typename Simd>
    inline constexpr std::size_t simdLanes<Simd, std::enable_if_t<isRecordRef<Simd>>> = []
    {
        using FRD = FlatRecordDim<typename Simd::AccessibleRecordDim>;
        using FirstFieldType = boost::mp11::mp_first<FRD>;
        static_assert(boost::mp11::mp_all_of_q<FRD, internal::SizeEqualTo<simdLanes<FirstFieldType>>>::value);
        return simdLanes<FirstFieldType>;
    }();

    /// Loads SIMD vectors of data starting from the given record reference to simd. Only field tags occurring in
    /// RecordRef are loaded. If Simd contains multiple fields of SIMD types, a SIMD vector will be fetched for each of
    /// the fields. The number of elements fetched per SIMD vector depends on the SIMD width of the vector. Simd is
    /// allowed to have different vector lengths per element.
    template<typename Simd, typename RecordRef>
    LLAMA_FN_HOST_ACC_INLINE void loadSimd(Simd& simd, const RecordRef& rr)
    {
        // structured simd type and record reference
        if constexpr(isRecordRef<Simd> && isRecordRef<RecordRef>)
        {
            using RecordDim = typename RecordRef::AccessibleRecordDim;
            forEachLeafCoord<RecordDim>(
                [&](auto rc) LLAMA_LAMBDA_INLINE
                {
                    using FieldType = GetType<RecordDim, decltype(rc)>;
                    using ElementSimd = std::decay_t<decltype(simd(rc))>;
                    using Traits = SimdTraits<ElementSimd>;

                    // TODO(bgruber): can we generalize the logic whether we can load a simd from that mapping?
                    if constexpr(mapping::isSoA<typename RecordRef::View::Mapping>)
                        simd(rc) = Traits::loadUnaligned(&rr(rc)); // SIMD load
                    // else if constexpr(mapping::isAoSoA<typename RecordRef::View::Mapping>)
                    //{
                    //    // it turns out we do not need the specialization, because clang already fuses the scalar
                    //    loads
                    //    // into a vector load :D
                    //    assert(rr.arrayDimsCoord()[0] % SIMD_WIDTH == 0);
                    //    // if(rr.arrayDimsCoord()[0] % SIMD_WIDTH != 0)
                    //    //    __builtin_unreachable(); // this also helps nothing
                    //    //__builtin_assume(rr.arrayDimsCoord()[0] % SIMD_WIDTH == 0);  // this also helps nothing
                    //    simd(rc) = Traits::load_from(&rr(rc)); // SIMD load
                    //}
                    else
                    {
                        auto b = ArrayIndexIterator{rr.view.mapping().extents(), rr.arrayIndex()};
                        for(auto i = 0; i < Traits::lanes; i++)
                            reinterpret_cast<FieldType*>(&simd(rc))[i]
                                = rr.view(*b++)(cat(typename RecordRef::BoundRecordCoord{}, rc)); // scalar loads
                    }
                });
        }
        // unstructured simd and reference type
        else if constexpr(!isRecordRef<Simd> && !isRecordRef<RecordRef>)
        {
            simd = SimdTraits<Simd>::loadUnaligned(&rr);
        }
        else
        {
            // TODO(bgruber): when could we get here? Is this always an error?
            static_assert(sizeof(Simd) == 0, "Invalid combination of Simd type and reference type");
        }
    }

    /// Stores SIMD vectors of element data from the given simd into memory starting at the provided record reference.
    /// Only field tags occurring in RecordRef are stored. If Simd contains multiple fields of SIMD types, a SIMD
    /// vector will be stored for each of the fields. The number of elements stored per SIMD vector depends on the
    /// SIMD width of the vector. Simd is allowed to have different vector lengths per element.
    template<typename RecordRef, typename Simd>
    LLAMA_FN_HOST_ACC_INLINE void storeSimd(RecordRef&& rr, Simd simd)
    {
        // structured Simd type and record reference
        if constexpr(isRecordRef<Simd> && isRecordRef<RecordRef>)
        {
            using RecordDim = typename RecordRef::AccessibleRecordDim;
            forEachLeafCoord<RecordDim>(
                [&](auto rc) LLAMA_LAMBDA_INLINE
                {
                    using FieldType = GetType<RecordDim, decltype(rc)>;
                    using ElementSimd = std::decay_t<decltype(simd(rc))>;
                    using Traits = SimdTraits<ElementSimd>;

                    // TODO(bgruber): can we generalize the logic whether we can store a simd to that mapping?
                    if constexpr(mapping::isSoA<typename RecordRef::View::Mapping>)
                        Traits::storeUnaligned(simd(rc), &rr(rc)); // SIMD store
                    else
                    {
                        // TODO(bgruber): how does this generalize conceptually to 2D and higher dimensions? in which
                        // direction should we collect SIMD values?
                        auto b = ArrayIndexIterator{rr.view.mapping().extents(), rr.arrayIndex()};
                        for(auto i = 0; i < Traits::lanes; i++)
                            rr.view (*b++)(cat(typename RecordRef::BoundRecordCoord{}, rc))
                                = reinterpret_cast<FieldType*>(&simd(rc))[i]; // scalar store
                    }
                });
        }
        // unstructured simd and reference type
        else if constexpr(!isRecordRef<Simd> && !isRecordRef<RecordRef>)
        {
            SimdTraits<Simd>::storeUnaligned(simd, &rr);
        }
        else
        {
            // TODO(bgruber): when could we get here? Is this always an error?
            static_assert(sizeof(Simd) == 0, "Invalid combination of Simd type and reference type");
        }
    }

    template<
        std::size_t N,
        template<typename, /* std::integral */ auto>
        typename MakeSizedSimd,
        typename View,
        typename UnarySimdFunction>
    void simdForEachN(View& view, UnarySimdFunction f)
    {
        using IndexType = typename View::Mapping::ArrayExtents::value_type;
        const auto total = product(view.mapping().extents());
        auto it = view.begin();
        IndexType i = 0;
        // simd loop
        while(i + IndexType{N} <= total)
        {
            SimdN<typename View::RecordDim, N, MakeSizedSimd> simd;
            loadSimd(simd, *it);
            if constexpr(std::is_void_v<decltype(f(simd))>)
                f(simd);
            else
                storeSimd(*it, f(simd));
            i += IndexType{N};
            it += IndexType{N};
        }
        // tail
        while(i < total)
        {
            auto scalar = One<typename View::RecordDim>{*it};
            if constexpr(std::is_void_v<decltype(f(scalar))>)
                f(scalar);
            else
                *it = f(scalar);
            ++i;
            ++it;
        }
    }

    template<
        template<typename>
        typename MakeLargestSimd,
        template<typename, /* std::integral */ auto>
        typename MakeSizedSimd,
        typename View,
        typename UnarySimdFunction>
    void simdForEach(View& view, UnarySimdFunction f)
    {
        constexpr auto n = llama::simdLanesFor<typename View::RecordDim, MakeLargestSimd>;
        simdForEachN<n, MakeSizedSimd>(view, f);
    }
} // namespace llama

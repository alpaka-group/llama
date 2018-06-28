#pragma once

namespace nbody
{

namespace allocator
{

namespace internal
{

template < typename T_Buffer >
struct AlpakaAccessor
{
    using PrimType = unsigned char;
    using BlobType = T_Buffer;

    AlpakaAccessor( BlobType buffer ) :
        buffer ( buffer )
    { }

    /* Explicit copy and move constructor and destructor definition because of
     * "calling a __host__ function from a __host__ __device__ function warnings
     * from nvidia compiler.
     */
    AlpakaAccessor( AlpakaAccessor const & ) = default;
    AlpakaAccessor( AlpakaAccessor && ) = default;
    ~AlpakaAccessor( ) = default;

    template< typename T_IndexType >
    auto
    operator[] ( T_IndexType && idx )
    -> PrimType &
    {
        return alpaka::mem::view::getPtrNative( buffer )[ idx ];
    }

    template< typename T_IndexType >
    auto operator[] ( T_IndexType && idx ) const
    -> const PrimType &
    {
        return alpaka::mem::view::getPtrNative( buffer )[ idx ];
    }

    BlobType buffer;
};

} // namespace internal

template<
    typename T_DevAcc,
    typename T_Dim,
    typename T_Size
>
struct Alpaka
{
    using DevAcc = T_DevAcc;
    using Dim = T_Dim;
    using Size = T_Size;

    using BufferType = alpaka::mem::buf::Buf<
        DevAcc,
        unsigned char,
        Dim,
        Size
    >;
    using BlobType = internal::AlpakaAccessor< BufferType >;
    using PrimType = typename BlobType::PrimType;
    using Parameter = DevAcc;

    static inline
    auto
    allocate(
        std::size_t count,
        Parameter devAcc
    )
    -> BlobType
    {
        BufferType buffer =
        alpaka::mem::buf::alloc<
            PrimType,
            Size
        > (
            devAcc,
            Size(count)
        );
        BlobType accessor( buffer );
        return accessor;
    }

};

template<
    typename T_DevAcc,
    typename T_Dim,
    typename T_Size,
    typename T_Mapping
>
struct AlpakaMirror
{
    using MirroredAllocator = Alpaka<
        T_DevAcc,
        T_Dim,
        T_Size
    >;
    using BlobType = typename MirroredAllocator::PrimType*;
    using PrimType = typename MirroredAllocator::PrimType;
    using MirroredView = llama::View<
        T_Mapping,
        typename MirroredAllocator::BlobType
    >;
    using Parameter = MirroredView;

    static inline
    auto
    allocate(
        std::size_t count,
        Parameter mirroredView
    )
    -> BlobType
    {
        return alpaka::mem::view::getPtrNative( mirroredView.blob[0].buffer );
    }
};

template<
    typename T_Acc,
    std::size_t T_count,
    std::size_t T_uniqueID
>
struct AlpakaShared
{
    using BlobType = unsigned char*;
    using PrimType = unsigned char;
    using Parameter = T_Acc;
    using AllocType = PrimType[ T_count ];

    static
    LLAMA_FN_HOST_ACC_INLINE
    auto
    allocate(
        std::size_t count,
        Parameter const & acc
    )
    -> BlobType
    {
        return alpaka::block::shared::st::allocVar<
            AllocType,
            T_uniqueID
        >( acc );

    }
};

} // namespace allocator

} // namespace nbody

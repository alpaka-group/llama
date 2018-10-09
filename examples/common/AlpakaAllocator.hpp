#pragma once

namespace common
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

/** Allocator to allocate memory for a \ref llama::View in the
 *  \ref llama::Factory using `alpaka::mem::buf::Buf` in the background. The
 *  view created with this allocator can only be used on the host side. For the
 *  use of the view on the device see \ref AlpakaMirror.
 * \tparam T_DevAcc alpaka `DevAcc`
 * \tparam T_Size alpaka size type
 * \see AlpakaMirror
 */
template<
    typename T_DevAcc,
    typename T_Size
>
struct Alpaka
{
    using DevAcc = T_DevAcc;
    using Size = T_Size;

    using BufferType = alpaka::mem::buf::Buf<
        DevAcc,
        unsigned char,
        alpaka::dim::DimInt< 1 >,
        Size
    >;
    /** blob type of this allocator is an internal accessor the the alpaka
     *  buffer
     */
    using BlobType = internal::AlpakaAccessor< BufferType >;
    /// primary type of this allocator is `unsigned char`
    using PrimType = typename BlobType::PrimType;
    /// the parameter is the alpaka `DevAcc` instance
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
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && BOOST_LANG_CUDA
        alpaka::mem::buf::pin( buffer );
#endif
        BlobType accessor( buffer );
        return accessor;
    }
};

/** Allocator to mirror the pointer of an \ref Alpaka allocated memory
 *  for a \ref llama::View in the \ref llama::Factory. The view created with
 *  this allocator can be used on the device side, but the memory is shared with
 *  the given view allocated with \ref Alpaka.
 * \tparam T_DevAcc alpaka `DevAcc`
 * \tparam T_Size alpaka size type
 * \tparam T_Mapping mapping used for creating the already existing view
 * \see Alpaka
 */
template<
    typename T_DevAcc,
    typename T_Size,
    typename T_Mapping
>
struct AlpakaMirror
{
    using MirroredAllocator = Alpaka<
        T_DevAcc,
        T_Size
    >;
    /// blob type of this allocator is `unsigned char*`
    using BlobType = typename MirroredAllocator::PrimType*;
    /// primary type of this allocator is `unsigned char`
    using PrimType = typename MirroredAllocator::PrimType;
    using MirroredView = llama::View<
        T_Mapping,
        typename MirroredAllocator::BlobType
    >;
    /// the parameter is the view which shall be mirrored
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

/** Allocator to allocate memory for a \ref llama::View in the
 *  \ref llama::Factory using alpaka shared memory in the background. The view
 *  created with this allocator can only be used on the view side as share
 *  memory exists only there.
 * \tparam T_Acc alpaka `Acc` type of the kernel
 * \tparam T_count Amount of memory needed in byte like needed for
 * \ref llama::allocator::Stack at compile time. In the future this may change
 *  and be a run time parameter (at least for dynamic shared memory allocation).
 * \tparam T_uniqueID at compile time unique ID needed by alpaka, best is to use
 *  e.g. `__COUNTER__` which is always unique as it increases after each use
 *  while preprocessing the code.
 * \see llama::allocator::Stack
 */
template<
    typename T_Acc,
    std::size_t T_count,
    std::size_t T_uniqueID
>
struct AlpakaShared
{
    /// blob type of this allocator is `unsigned char*`
    using BlobType = unsigned char*;
    /// primary type of this allocator is `unsigned char`
    using PrimType = unsigned char;
    /// the allocation parameter is the alpaka `Acc` type of the kernel
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

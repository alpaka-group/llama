/* Copyright 2018 Alexander Matthes
 *
 * This file is part of LLAMA.
 *
 * LLAMA is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * LLAMA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with LLAMA.  If not, see <www.gnu.org/licenses/>.
 */

#pragma once

#include "TreeElement.hpp"

namespace llama
{

namespace mapping
{

namespace tree
{

template<
	typename T_Tree,
	template< class, class > class T_InnerOp,
	template< class, class > class T_OuterOp,
	template< class, class > class T_LeafFunctor,
	typename T_SFINAE = void
>
struct Reduce;

namespace internal
{

// Leaf
template<
	typename T_Tree,
	template< class, class > class T_InnerOp,
	template< class, class > class T_OuterOp,
	template< class, class > class T_LeafFunctor,
	typename T_SFINAE = void
>
struct ReduceElementType
{
	LLAMA_FN_HOST_ACC_INLINE
	auto
	operator()( decltype( T_Tree::count ) const & count ) const
	-> std::size_t
	{
		return T_LeafFunctor<
			typename T_Tree::Type,
			decltype( T_Tree::count )
		>()( count );
	}
};

// Node
template<
	typename T_Tree,
	template< class, class > class T_InnerOp,
	template< class, class > class T_OuterOp,
	template< class, class > class T_LeafFunctor
>
struct ReduceElementType<
	T_Tree,
	T_InnerOp,
	T_OuterOp,
	T_LeafFunctor,
	typename std::enable_if<
		( SizeOfTuple< typename T_Tree::Type >::value > 1 )
	>::type
>
{
	using IterTree = typename TreePopFrontChild< T_Tree >::ResultType;

	LLAMA_FN_HOST_ACC_INLINE
	auto
	operator()(
        typename T_Tree::Type const & childs,
        decltype( T_Tree::count ) const & count
    ) const
	-> std::size_t
	{
		return T_InnerOp<
			decltype(
				Reduce<
					typename T_Tree::Type::FirstElement,
					T_InnerOp,
					T_OuterOp,
					T_LeafFunctor
				>()( childs.first )
			),
			decltype(
				internal::ReduceElementType<
					IterTree,
					T_InnerOp,
					T_OuterOp,
					T_LeafFunctor
				>()(
					childs.rest,
					count
				)
			)
		>::apply(
			Reduce<
				typename T_Tree::Type::FirstElement,
				T_InnerOp,
				T_OuterOp,
				T_LeafFunctor
			>()( childs.first ),
			internal::ReduceElementType<
				IterTree,
				T_InnerOp,
				T_OuterOp,
				T_LeafFunctor
			>()(
				childs.rest,
				count
			)
		);
	}
};

// Node with one (last) child
template<
	typename T_Tree,
	template< class, class > class T_InnerOp,
	template< class, class > class T_OuterOp,
	template< class, class > class T_LeafFunctor
>
struct ReduceElementType<
	T_Tree,
	T_InnerOp,
	T_OuterOp,
	T_LeafFunctor,
	typename std::enable_if<
		SizeOfTuple< typename T_Tree::Type >::value == 1
	>::type
>
{
	LLAMA_FN_HOST_ACC_INLINE
	auto
	operator()(
        typename T_Tree::Type const & childs,
        decltype( T_Tree::count ) const & count
    ) const
	-> std::size_t
	{
		return Reduce<
			typename T_Tree::Type::FirstElement,
			T_InnerOp,
			T_OuterOp,
			T_LeafFunctor
		>()( childs.first );
	}
};

} //namespace internal

template<
	typename T_Tree,
	template< class, class > class T_InnerOp,
	template< class, class > class T_OuterOp,
	template< class, class > class T_LeafFunctor,
	typename T_SFINAE
>
struct Reduce
{
	LLAMA_FN_HOST_ACC_INLINE
	auto
	operator()(
		typename T_Tree::Type const & childs,
		decltype( T_Tree::count ) const & count
	) const
	-> std::size_t
	{
		return T_OuterOp<
			decltype( T_Tree::count ),
			decltype(
				internal::ReduceElementType<
					T_Tree,
					T_InnerOp,
					T_OuterOp,
					T_LeafFunctor
				>()(
					childs,
					count
				)
			)
		>::apply(
			count,
			internal::ReduceElementType<
				T_Tree,
				T_InnerOp,
				T_OuterOp,
				T_LeafFunctor
			>()(
				childs,
				count
			)
		);
	}

	LLAMA_FN_HOST_ACC_INLINE
	auto
	operator()( T_Tree const & tree ) const
	-> std::size_t
	{
		return operator()(
			tree.childs,
			// cuda doesn't like references to static members of they are
			// not defined somewhere although only type informations
			// are used which is the case for runtime=std::integral_constant
			LLAMA_DEREFERENCE( tree.count )
		);
	}
};

template<
	typename T_Tree,
	template< class, class > class T_InnerOp,
	template< class, class > class T_OuterOp,
	template< class, class > class T_LeafFunctor
>
struct Reduce<
	T_Tree,
	T_InnerOp,
	T_OuterOp,
	T_LeafFunctor,
	typename T_Tree::IsTreeElementWithoutChilds
>
{
	LLAMA_FN_HOST_ACC_INLINE
	auto
	operator()( decltype( T_Tree::count ) const & count ) const
	-> std::size_t
	{
		return T_OuterOp<
			decltype( T_Tree::count ),
			decltype(
				internal::ReduceElementType<
					T_Tree,
					T_InnerOp,
					T_OuterOp,
					T_LeafFunctor
				>()( count )
			)
		>::apply(
			count,
			internal::ReduceElementType<
				T_Tree,
				T_InnerOp,
				T_OuterOp,
				T_LeafFunctor
			>()( count )
		);
	}

	LLAMA_FN_HOST_ACC_INLINE
	auto
	operator()( T_Tree const & tree ) const
	-> std::size_t
	{
		// cuda doesn't like references to static members of they are
		// not defined somewhere although only type informations
		// are used which is the case for runtime=std::integral_constant
		//~ return operator()( (tree.count) );
		return operator()( LLAMA_DEREFERENCE( tree.count ) );
	}
};

} // namespace tree

} // namespace mapping

} // namespace llama


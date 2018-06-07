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
	template< class, class > class T_LeaveFunctor,
	typename T_SFINAE = void
>
struct Reduce;

namespace internal
{

// Leave
template<
	typename T_Tree,
	template< class, class > class T_InnerOp,
	template< class, class > class T_OuterOp,
	template< class, class > class T_LeaveFunctor,
	typename T_SFINAE = void
>
struct ReduceElementType
{
	LLAMA_FN_HOST_ACC_INLINE
	auto
	operator()( decltype( T_Tree::count ) const & count ) const
	-> std::size_t
	{
		return T_LeaveFunctor<
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
	template< class, class > class T_LeaveFunctor
>
struct ReduceElementType<
	T_Tree,
	T_InnerOp,
	T_OuterOp,
	T_LeaveFunctor,
	typename std::enable_if<
		( TupleLength< typename T_Tree::Type >::value > 1 )
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
					T_LeaveFunctor
				>()( childs.first )
			),
			decltype(
				internal::ReduceElementType<
					IterTree,
					T_InnerOp,
					T_OuterOp,
					T_LeaveFunctor
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
				T_LeaveFunctor
			>()( childs.first ),
			internal::ReduceElementType<
				IterTree,
				T_InnerOp,
				T_OuterOp,
				T_LeaveFunctor
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
	template< class, class > class T_LeaveFunctor
>
struct ReduceElementType<
	T_Tree,
	T_InnerOp,
	T_OuterOp,
	T_LeaveFunctor,
	typename std::enable_if<
		TupleLength< typename T_Tree::Type >::value == 1
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
			T_LeaveFunctor
		>()( childs.first );
	}
};

} //namespace internal

template<
	typename T_Tree,
	template< class, class > class T_InnerOp,
	template< class, class > class T_OuterOp,
	template< class, class > class T_LeaveFunctor,
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
					T_LeaveFunctor
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
				T_LeaveFunctor
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
			decltype(tree.count)( tree.count )
		);
	}
};

template<
	typename T_Tree,
	template< class, class > class T_InnerOp,
	template< class, class > class T_OuterOp,
	template< class, class > class T_LeaveFunctor
>
struct Reduce<
	T_Tree,
	T_InnerOp,
	T_OuterOp,
	T_LeaveFunctor,
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
					T_LeaveFunctor
				>()( count )
			)
		>::apply(
			count,
			internal::ReduceElementType<
				T_Tree,
				T_InnerOp,
				T_OuterOp,
				T_LeaveFunctor
			>()( count )
		);
	}

	LLAMA_FN_HOST_ACC_INLINE
	auto
	operator()( T_Tree const & tree ) const
	-> std::size_t
	{
		return operator()( tree.count );
	}
};

} // namespace tree

} // namespace mapping

} // namespace llama


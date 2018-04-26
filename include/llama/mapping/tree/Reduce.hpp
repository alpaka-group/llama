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
	typename T_InnerOp,
	typename T_OuterOp,
	template< class > class T_LeaveFunctor
>
struct Reduce;

namespace internal
{

// Leave
template<
	typename T_Tree,
	typename T_InnerOp,
	typename T_OuterOp,
	template< class > class T_LeaveFunctor,
	typename T_SFINAE = void
>
struct ReduceElementType
{
	auto
	operator()( const T_Tree leave )
	-> std::size_t
	{
		return T_LeaveFunctor< T_Tree >()( leave );
	}
};

// Node
template<
	typename T_Tree,
	typename T_InnerOp,
	typename T_OuterOp,
	template< class > class T_LeaveFunctor
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
	using IterTree = TreeElement<
		typename T_Tree::Identifier,
		typename T_Tree::Type::RestTuple
	>;
	auto
	operator()( const T_Tree tree )
	-> std::size_t
	{
		return T_InnerOp::apply(
			Reduce<
				typename T_Tree::Type::FirstElement,
				T_InnerOp,
				T_OuterOp,
				T_LeaveFunctor
			>()( tree.childs.first ),
			internal::ReduceElementType<
				IterTree,
				T_InnerOp,
				T_OuterOp,
				T_LeaveFunctor
			>()(
				IterTree(
					tree.count,
					tree.childs.rest
				)
			)
		);
	}
};

// Node with one (last) child
template<
	typename T_Tree,
	typename T_InnerOp,
	typename T_OuterOp,
	template< class > class T_LeaveFunctor
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
	auto
	operator()( const T_Tree tree )
	-> std::size_t
	{
		return Reduce<
			typename T_Tree::Type::FirstElement,
			T_InnerOp,
			T_OuterOp,
			T_LeaveFunctor
		>()( tree.childs.first );
	}
};

} //namespace internal

template<
	typename T_Tree,
	typename T_InnerOp,
	typename T_OuterOp,
	template< class > class T_LeaveFunctor
>
struct Reduce
{
	auto
	operator()( const T_Tree tree )
	-> std::size_t
	{
		return T_OuterOp::apply(
			tree.count,
			internal::ReduceElementType<
				T_Tree,
				T_InnerOp,
				T_OuterOp,
				T_LeaveFunctor
			>()( tree )
		);
	}
};

} // namespace tree

} // namespace mapping

} // namespace llama


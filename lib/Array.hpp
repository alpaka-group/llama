#pragma once

namespace llama
{

template <typename T,size_t dim>
struct Array
{
	static constexpr size_t count = dim;
	T element[dim];
	template <typename IndexType>
	T& operator[] (IndexType&& idx)
	{
		return element[idx];
	}
	template <typename IndexType>
	const T& operator[] (IndexType&& idx) const
	{
		return element[idx];
	}
	Array<T,count-1> pop_front() const
	{
		Array<T,count-1> result;
		for (size_t i = 0; i < count-1; i++)
			result.element[i] = element[i+1];
		return result;
	}
	Array<T,count-1> pop_back() const
	{
		Array<T,count-1> result;
		for (size_t i = 0; i < count-1; i++)
			result.element[i] = element[i];
		return result;
	}
	Array<T,count+1> push_front(T const new_element) const
	{
		Array<T,count+1> result;
		for (size_t i = 0; i < count-1; i++)
			result.element[i+1] = element[i];
		result.element[0] = new_element;
		return result;
	}
	Array<T,count+1> push_back(T const new_element) const
	{
		Array<T,count+1> result;
		for (size_t i = 0; i < count-1; i++)
			result.element[i] = element[i];
		result.element[count] = new_element;
		return result;
	}
};

} //namespace llama

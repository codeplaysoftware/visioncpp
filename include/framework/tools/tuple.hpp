// This file is part of VisionCpp, a lightweight C++ template library
// for computer vision and image processing.
//
// Copyright (C) 2016 Codeplay Software Limited. All Rights Reserved.
//
// Contact: visioncpp@codeplay.com
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// \file tuple.hpp
/// \brief Tuples standard layout implementation.
/// http://en.cppreference.com/w/cpp/utility/tuple

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_TOOLS_TUPLE_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_TOOLS_TUPLE_HPP_

namespace visioncpp {
namespace internal {
namespace tools {
/// \brief Contains standard layout tuple implementation.
namespace tuple {
/// \struct EnableIf
/// \brief The EnableIf struct is used to statically define type based on the
/// condition.
template <bool, typename T = void>
struct EnableIf {};
/// \brief Specialisation of the \ref EnableIf when the condition is matched.
template <typename T>
struct EnableIf<true, T> {
  typedef T type;
};

/// \struct Tuple
/// \brief The tuple is a fixed-size collection of heterogeneous values.
/// \tparam Ts...	-	the types of the elements that the tuple stores.
/// Empty list is supported.
template <class... Ts>
struct Tuple {};

/// \brief Specialisation of the \ref Tuple class when it has at least one
/// element.
/// \tparam t : the type of the first element in the tuple.
/// \tparam ts... the rest of the elements in the tuple. Ts... can be empty.
template <class T, class... Ts>
struct Tuple<T, Ts...> {
  Tuple(T t, Ts... ts) : head(t), tail(ts...) {}

  T head;
  Tuple<Ts...> tail;
};

/// \struct ElemTypeHolder
/// \brief ElemTypeHolder class is used to specify the types of the
/// elements inside the tuple.
/// \tparam size_t The number of elements inside the tuple.
/// \tparam class The tuple class.
template <size_t, class>
struct ElemTypeHolder;

/// \brief Specialisation of the \ref ElemTypeHolder class when the number of
/// the
/// elements inside the tuple is 1.
template <class T, class... Ts>
struct ElemTypeHolder<0, Tuple<T, Ts...>> {
  typedef T type;
};

/// \brief Specialisation of the \ref ElemTypeHolder class when the number of
/// the
/// elements inside the tuple is bigger than 1. It recursively calls itself to
/// detect the type of each element in the tuple.
/// \tparam T : the type of the first element in the tuple.
/// \tparam Ts... the rest of the elements in the tuple. Ts... can be empty.
/// \tparam K is the Kth element in the tuple.
template <size_t k, class T, class... Ts>
struct ElemTypeHolder<k, Tuple<T, Ts...>> {
  typedef typename ElemTypeHolder<k - 1, Tuple<Ts...>>::type type;
};

/// get
/// \brief Extracts the first element from the tuple.
/// K=0 represents the first element of the tuple. The tuple cannot be empty.
/// \tparam Ts... are the elements type in the tuple.
/// \param t is the tuple whose contents to extract.
/// \return  typename ElemTypeHolder<0, Tuple<Ts...>>\::type &>\::type
template <size_t k, class... Ts>
typename EnableIf<k == 0,
                  typename ElemTypeHolder<0, Tuple<Ts...>>::type &>::type
get(Tuple<Ts...> &t) {
  return t.head;
}

/// get
/// \brief Extracts the Kth element from the tuple.
/// \tparam K is an integer value in [0,sizeof...(Types)) range.
/// \tparam T is the (sizeof...(Types) -(K+1)) element in the tuple.
/// \tparam Ts... are the elements type in the tuple.
/// \param t is the tuple whose contents to extract.
/// \return  typename ElemTypeHolder<K, Tuple<Ts...>>\::type &>\::type
template <size_t k, class T, class... Ts>
typename EnableIf<k != 0,
                  typename ElemTypeHolder<k, Tuple<T, Ts...>>::type &>::type
get(Tuple<T, Ts...> &t) {
  return get<k - 1>(t.tail);
}

/// get
/// \brief Extracts the first element from the tuple when the tuple and all the
/// elements inside it are const.
/// K=0 represents the first element of the tuple. The tuple cannot be empty.
/// \tparam Ts... are the elements type in the tuple.
/// \param t is the const tuple whose contents to extract.
/// \return  const typename ElemTypeHolder<0, Tuple<Ts...>>\::type &>\::type
template <size_t k, class... Ts>
typename EnableIf<k == 0,
                  const typename ElemTypeHolder<0, Tuple<Ts...>>::type &>::type
get(const Tuple<Ts...> &t) {
  return t.head;
}

/// get
/// \brief Extracts the Kth element from the tuple when the tuple and all the
/// elements inside are const.
/// \tparam K is an integer value in [0,sizeof...(Types)) range.
/// \tparam T is the (sizeof...(Types) -(K+1)) element in the tuple.
/// \tparam Ts... are the elements type in the tuple.
/// \param t is the const tuple whose contents to extract.
/// \return  const typename ElemTypeHolder<K, Tuple<Ts...>>\::type &>\::type
template <size_t k, class T, class... Ts>
typename EnableIf<
    k != 0, const typename ElemTypeHolder<k, Tuple<T, Ts...>>::type &>::type
get(const Tuple<T, Ts...> &t) {
  return get<k - 1>(t.tail);
}

/// make_tuple
/// \brief Creates a tuple object, deducing the target type from the types of
/// the arguments.
/// \tparam Args the type of the arguments to construct the tuple from.
/// \param args zero or more arguments to construct the tuple from.
/// \return Tuple<Args...>
template <typename... Args>
Tuple<Args...> make_tuple(Args... args) {
  return Tuple<Args...>(args...);
}

/// size
/// \brief Returns the number of the elements in the tuple as a
/// compile-time expression.
/// \tparam Args the type of the arguments to construct the tuple from.
/// \return size_t
template <typename... Args>
static constexpr size_t size(Tuple<Args...> &) {
  return sizeof...(Args);
}

/// \struct Index_list
/// \brief Creates a list of indices created for the elements in the tuple.
/// \tparam Is... a list of indices from range of [0 to sizeof...(tuple
/// elements)).
template <size_t... Is>
struct Index_list {};

/// \struct RangeBuilder
/// \brief Collects internal details for index ranges generation [MIN, MAX).
/// Declares primary template for the RangeBuilder.
/// \tparam MIN is the starting index in the tuple.
/// \tparam N represents sizeof..(elemens)- sizeof...(Is)
/// \tparam Is... Collection of so far generated indices.
template <size_t MIN, size_t N, size_t... Is>
struct RangeBuilder;

/// \brief Specialisation of the \ref RangeBuilder when the
/// MIN==MAX. In this case the Is... contains [0 to sizeof...(tuple elements))
/// indices.
/// \tparam MIN is the starting index of the tuple.
/// \tparam Is is collection of [0 to sizeof...(tuple elements)) indices.
template <size_t MIN, size_t... Is>
struct RangeBuilder<MIN, MIN, Is...> {
  typedef Index_list<Is...> type;
};

/// \brief Specialisation of the RangeBuilder class when N!=MIN.
/// In this case we are recursively subtracting the N by one and adding one
/// index to the Is... list until MIN==N.
/// \tparam MIN is the starting index in the tuple.
/// \tparam N represents reduction value from MAX which is equal to
/// sizeof..(elemens)- sizeof...(Is).
/// \tparam Is... Collection of so far generated indices.
template <size_t MIN, size_t N, size_t... Is>
struct RangeBuilder : public RangeBuilder<MIN, N - 1, N - 1, Is...> {};

/// \brief IndexRange that returns a index from range of [MIN, MAX).
/// \tparam MIN is the starting index in the tuple.
/// \tparam MAX is the size of the tuple.
template <size_t MIN, size_t MAX>
using Index_range = typename RangeBuilder<MIN, MAX>::type;

/// append_impl
/// \brief Unpacks the elements of the input tuple t and creates a new tuple
/// by adding the element a at the end of it.
/// \tparam Args... the type of the elements inside the tuple t.
/// \tparam T the type of the new element that is going to be added at the end
/// of the returned tuple.
/// \tparam I... is the list of indices from [0 to sizeof...(t)) range.
/// \param t the tuple on which we want to append a.
/// \param a the new elements that are going to be added to the returned tuple.
/// \return Tuple<Args..., T>
template <typename... Args, typename T, size_t... I>
Tuple<Args..., T> append_impl(internal::tools::tuple::Tuple<Args...> t, T a,
                              internal::tools::tuple::Index_list<I...>) {
  return internal::tools::tuple::make_tuple(get<I>(t)..., a);
}

/// append
/// \brief the deduction function for \ref append_impl that automatically
/// generates the \ref Index_range.
/// \tparam Args... the type of the elements inside the tuple t.
/// \tparam T the type of the new element that is going to be added at the end
/// of the returned tuple.
/// \param t the tuple on which we want to append a.
/// \param a the new elements that are going to be added to the returned tuple.
/// \return Tuple<Args..., T>
template <typename... Args, typename T>
Tuple<Args..., T> append(Tuple<Args...> t, T a) {
  return internal::tools::tuple::append_impl(
      t, a, internal::tools::tuple::Index_range<0, sizeof...(Args)>());
}

/// append_impl
/// \brief This is a specialisation of the \ref append_impl for the
/// concatenation
/// of the t2 tupe at the end of the t1 tuple. Both tuples are unpacked.
/// Index_range is generated for each of them and output tuple T is created.
/// The return tuple contains both elements of t1 and t2 tuples.
/// \tparam Args1... The type of the t1 tuple.
/// \tparam Args2... The type of the t2 tuple.
/// \tparam I1... The list of the indices from [0 to sizeof...(t1)) range.
/// \tparam I2... The list of the indices from [0 to sizeof...(t2)) range.
/// \param t1 Is the tuple on which we want to append the t2 tuple.
/// \param t2 Is the tuple to be appended.
/// \return Tuple<Args1..., Args2...>
template <typename... Args1, typename... Args2, size_t... I1, size_t... I2>
Tuple<Args1..., Args2...> append_impl(
    internal::tools::tuple::Tuple<Args1...> t1,
    internal::tools::tuple::Tuple<Args2...> t2,
    internal::tools::tuple::Index_list<I1...>,
    internal::tools::tuple::Index_list<I2...>) {
  return internal::tools::tuple::make_tuple(
      internal::tools::tuple::get<I1>(t1)...,
      internal::tools::tuple::get<I2>(t2)...);
}

/// append
/// \brief Deduction function of the \ref append_impl for the append of the
/// tuple
/// t2 to the t1 tuple. The \ref Index_range for both tuple is automatically
/// generated.
/// \tparam Args1... The type of the t1 tuple.
/// \tparam Args2... The type of the t2 tuple.
/// \param t1 Is the tuple on which we want to append the t2 tuple.
/// \param t2 Is the tuple to be appended.
/// \return Tuple<Args1..., Args2...>
template <typename... Args1, typename... Args2>
Tuple<Args1..., Args2...> append(internal::tools::tuple::Tuple<Args1...> t1,
                                 internal::tools::tuple::Tuple<Args2...> t2) {
  return internal::tools::tuple::append_impl(
      t1, t2, internal::tools::tuple::Index_range<0, sizeof...(Args1)>(),
      internal::tools::tuple::Index_range<0, sizeof...(Args2)>());
}
}  // tuple
}  // tools
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_TOOLS_TUPLE_HPP_

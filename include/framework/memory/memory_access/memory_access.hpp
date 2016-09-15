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

/// \file memory_access.hpp
/// this file contains a set of forward declarations and include headers
/// required for constructing and accessing memory on both host and device.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEMORY_ACCESS_MEMORY_ACCESS_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEMORY_ACCESS_MEMORY_ACCESS_HPP_

namespace visioncpp {
namespace internal {
/// \struct CompareIdBasedScope
/// \brief this is used for range check to make sure the
/// index is within the range. It uses the local value when the Conds
/// are true.
/// template parameters:
/// \tparam Conds: determines whether or not the local variable should be used
/// \tparam LDSize : determines the local dimension size
/// \tparam GDSize: determines the global dimension size
/// \tparam T determines the type of the dimension index
template <bool Conds, size_t LDSize, size_t GDSize, typename T>
struct CompareIdBasedScope {
  /// function get
  /// \brief returns the local range check:
  /// \param l is the local dimension size
  /// \param g is the global dimension size
  /// \param i is the offset needed to be added to the local dimension
  /// before comparison
  /// \return bool
  static inline bool get(T &l, int &i, T &g) { return (l + i < LDSize); }
};

/// \brief specialisation of the CompareIdBasedScope when the Conds is false in
/// this case the range check is with the global size
/// template parameters:
/// \tparam LDSize : determines the local dimension size
/// \tparam GDSize: determines the global dimension size
/// \tparam T: determines the type of the dimension index
template <size_t LDSize, size_t GDSize, typename T>
struct CompareIdBasedScope<false, LDSize, GDSize, T> {
  /// function get
  /// \brief returns the global range check:
  /// \param l is the local dimension size
  /// \param g is the global dimension size
  /// \param i is the offset needed to be added to the global dimension
  /// before comparison
  /// return bool
  static inline bool get(T &l, int &i, T &g) {
    return ((l + i < LDSize) && (g + i < GDSize));
  }
};
/// function get_compare
/// template deduction for CompareIdBasedScope
/// template parameters:
/// \tparam Conds: determines whether or not the local variable should be used
/// \tparam LDSize : determines the local dimension size
/// \tparam GDSize: determines the global dimension size
/// \tparam T: determines the type of the dimension index
/// function parameters:
/// \param l is the local dimension size
/// \param g is the global dimension size
/// \param i is the offset needed to be added to the correct dimension
/// before comparison
/// return bool
template <bool Conds, size_t LDSize, size_t GDSize, typename T>
static inline bool get_compare(T l, int i, T g) {
  return CompareIdBasedScope<Conds, LDSize, GDSize, T>::get(l, i, g);
}

/// \struct GetIdBasedScope
/// brief This is used  to get the correct Id based on the condition. True means
/// local
/// template parameters
/// \tparam Conds: determines whether or not the local variable should be used
/// \tparam T: determines the type of the dimension index
template <bool Conds, typename T>
struct GetIdBasedScope {
  /// function get
  /// \brief returns the local range check:
  /// \param l is the local dimension size
  /// \param g is the global dimension size
  /// \return T
  static inline T get(T l, T g) { return l; }
};
/// \brief specialisation of the GetIdBasedScope when the condition is false. In
/// this case the get function return global index.
template <typename T>
struct GetIdBasedScope<false, T> {
  /// function get
  /// \brief returns the local range check:
  /// \param l is the local dimension size
  /// \param g is the global dimension size
  /// \return T
  static inline T get(T l, T g) { return g; }
};

/// function id_val
/// \brief template deduction for GetIdBasedScope.
/// template parameters
/// \tparam Conds: determines whether or not the local variable should be used
/// \tparam T: determines the type of the dimension index
/// function parameters:
/// \param l is the local dimension size
/// \param g is the global dimension size
/// \return T
template <bool Conds, typename T>
static inline T id_val(T l, T g) {
  return GetIdBasedScope<Conds, T>::get(l, g);
}

/// \struct MemoryTrait
/// \brief This class is used to determine the ElementType of accessor
/// template parameters
/// \tparam LeafType: the type of the memory
/// \tparam T: the element type
template <size_t LeafType, typename T>
struct MemoryTrait {
  using Type = typename tools::RemoveAll<T>::Type::value_type;
};
/// \brief specialisation of MemoryTrait when the LeafType is Host
template <typename T>
struct MemoryTrait<memory_type::Host, T> {
  using Type = typename tools::RemoveAll<T>::Type;
};

/// function calculate_index
/// \brief this function is used to calculate the index access of the memory
/// pointer on the device.
/// parameters:
/// \param c :  column index
/// \param r : row index
/// \param cols :  column dimension
/// \param rows :  row dimension
/// \return size_t
static inline size_t calculate_index(size_t c, size_t r, size_t cols,
                                     size_t rows) {
  return (((r * cols) + c) < cols * rows) ? ((r * cols) + c)
                                          : (cols * rows) - 1;
}
}  // internal
}  // visioncpp
#include "mem_coordinate.hpp"
#include "mem_neighbour.hpp"
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEMORY_ACCESS_MEMORY_ACCESS_HPP_

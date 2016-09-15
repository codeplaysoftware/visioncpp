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

/// \file mem_neighbour.hpp
/// this file contains different types of memory required for executing
/// neighbour operation to calculate each pixel output

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEMORY_ACCESS_MEM_NEIGHBOUR_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEMORY_ACCESS_MEM_NEIGHBOUR_HPP_

namespace visioncpp {
namespace internal {
/// \struct LocalNeighbour
/// \brief LocalNeighbour is used to provide local access for each
/// element of the local memory based on the Coordinate passed by eval
/// expression. It is used as an input type for user functor when the local
/// neighbour operation is required.
/// template parameters
/// \tparam T is the pixel type for the local memory
template <typename T>
struct LocalNeighbour {
 public:
  using PixelType = T;
  cl::sycl::local_ptr<T> &ptr;

  int I_c;
  int I_r;
  LocalNeighbour(cl::sycl::local_ptr<T> &ptr, size_t colsArg, size_t rowsArg)
      : ptr(ptr), I_c(0), I_r(0), cols(colsArg), rows(rowsArg) {}
  /// function set_offset:
  /// \brief used to  set the local memory offset for each local thread
  /// function parameters:
  /// \param c : column index
  /// \param r: row index
  /// \return void
  inline void set_offset(int c, int r) {
    I_c = c;
    I_r = r;
  }
  /// function at provides access to a specific Coordinate for a 2d buffer
  /// parameters:
  /// \param c: column index
  /// \param r: row index
  /// \return PixelType
  inline PixelType at(int c, int r) const {
    c = (c >= 0 ? c : 0);
    r = (r >= 0 ? r : 0);
    return ptr[calculate_index(c, r, cols, rows)];
  }
  /// function at provides access to a specific Coordinate for a 1d buffer
  /// parameters:
  /// \param c: index
  /// \return PixelType
  inline PixelType at(int c) const { return ptr[c]; }

 private:
  size_t cols;
  size_t rows;
};
/// \struct GlobalNeighbour
/// \brief GlobalNeighbour is used to provide local access for each
/// element of the global memory based on the Coordinate passed by eval
/// expression. It is used as an input type for user functor when the global
/// neighbour operation is required.
/// template parameters
/// \tparam T is the pixel type for the global memory
template <typename T>
struct GlobalNeighbour {
  using PixelType = T;
  size_t I_c;
  size_t I_r;
  size_t cols;
  size_t rows;
  cl::sycl::global_ptr<T> &ptr;
  GlobalNeighbour(cl::sycl::global_ptr<T> &ptr, size_t colsArg, size_t rowsArg)
      : I_c(0), I_r(0), cols(colsArg), rows(rowsArg), ptr(ptr) {}
  /// function set_offset:
  /// \brief used to set the global memory offset for each global thread
  /// function parameters:
  /// \param c : column index
  /// \param r: row index
  /// \return void
  inline void set_offset(int c, int r) {
    I_c = c;
    I_r = r;
  }
  /// function at provides access to a specific Coordinate for a 2d buffer
  /// parameters:
  /// \param c: column index
  /// \param r: row index
  /// \return PixelType
  inline PixelType at(int c, int r) const {
    c = (c >= 0 ? c : 0);
    r = (r >= 0 ? r : 0);
    return ptr[calculate_index(c, r, cols, rows)];
  }
  /// function at provides access to a specific coordinate for a 1d buffer
  /// parameters:
  /// \param c:  index
  /// \return PixelType
  inline PixelType at(int c) const { return ptr[c]; }
};
/// \struct ConstNeighbour
/// \brief ConstNeighbour is used to provide global access to the constant
/// memory. It is used as an input type for user functor when a constant pointer
/// needed to be passed on the device side. An example of such node can be a
/// filter node for convolution operation.
/// template parameters
/// \tparam T is the pixel type for the constant memory
template <typename T>
struct ConstNeighbour {
  using PixelType = T;
  cl::sycl::constant_ptr<T> &ptr;
  size_t cols;
  size_t rows;
  ConstNeighbour(cl::sycl::constant_ptr<T> &ptr, size_t colsArg, size_t rowsArg)
      : ptr(ptr), cols(colsArg), rows(rowsArg) {}
  /// function at provides access to a specific coordinate for a 2d buffer
  /// parameters:
  /// \param c: column index
  /// \param r: row index
  /// \return PixelType
  inline PixelType at(int c, int r) const {
    c = (c >= 0 ? c : 0);
    r = (r >= 0 ? r : 0);
    return ptr[calculate_index(c, r, cols, rows)];
  }
  /// function at provides access to an specific Coordinate for a 1d buffer
  /// parameters:
  /// \param c:  index
  /// \return PixelType
  inline PixelType at(int c) const { return ptr[c]; }
};
}  // memory
}  // internal
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEMORY_ACCESS_MEM_NEIGHBOUR_HPP_
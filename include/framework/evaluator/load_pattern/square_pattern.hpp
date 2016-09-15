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

/// \file square_pattern.hpp
/// \brief this file contains the partial specialisation of the Fill for
/// LeafNode

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_LOAD_PATTERN_SQUARE_PATTERN_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_LOAD_PATTERN_SQUARE_PATTERN_HPP_

namespace visioncpp {
namespace internal {
/// \brief Partial specialisation of the Fill when the LeafNode contains the
/// const variable. In this case we load nothing in to the shared memory as
/// there is no shared memory for const variable and the const variable directly
/// copied to the device and accessed by each thread.
template <size_t N, size_t Rows, size_t Cols, size_t Sc, size_t LVL,
          typename Loc, typename... Params>
struct Fill<LeafNode<PlaceHolder<memory_type::Const, N, Cols, Rows, Sc>, LVL>,
            Loc, Params...> {
  template <size_t Halo_Top, size_t Halo_Left, size_t Halo_Butt,
            size_t Halo_Right, size_t Offset, size_t LC, size_t LR>
  static void fill_neighbour(Loc &cOffset,
                             const tools::tuple::Tuple<Params...> &t) {
    // no need to do anything the memory is read only
  }
};
/// \brief Partial specialisation of the Fill when the LeafNode contains a sycl
/// buffer created on constant memory. In this case we load nothing in to the
/// shared memory as there is no shared memory created for a buffer on a device
/// constant memory. Such a buffer is directly accessed on the device by each
/// thread.
template <size_t Memory_Type, size_t N, size_t Rows, size_t Cols, size_t LVL,
          typename Loc, typename... Params>
struct Fill<
    LeafNode<PlaceHolder<Memory_Type, N, Cols, Rows, scope::Constant>, LVL>,
    Loc, Params...> {
  template <size_t Halo_Top, size_t Halo_Left, size_t Halo_Butt,
            size_t Halo_Right, size_t Offset, size_t LC, size_t LR>
  static void fill_neighbour(Loc &cOffset,
                             const tools::tuple::Tuple<Params...> &t) {}
};
/// \brief Partial specialisation of the Fill when the LeafNode contains the
/// sycl buffer on the global memory. In this case each work group loads a
/// rectangle block of (LR,LC) in to their dedicated local memory.
template <size_t Memory_Type, size_t N, size_t Rows, size_t Cols, size_t LVL,
          size_t Sc, typename Loc, typename... Params>
struct Fill<LeafNode<PlaceHolder<Memory_Type, N, Cols, Rows, Sc>, LVL>, Loc,
            Params...> {
  template <size_t Halo_Top, size_t Halo_Left, size_t Halo_Butt,
            size_t Halo_Right, size_t Index, size_t LC, size_t LR>
  static void fill_neighbour(Loc &cOffset,
                             const tools::tuple::Tuple<Params...> &t) {
    static_assert(Cols > 0 && LC > 0, "Cols must be greater than 0");
    static_assert(Rows > 0 && LR > 0, "Rows must be greater than 0");
    // calculation begins from here for unary op
    for (int i = 0; i < LC; i += cOffset.cLRng) {
      if ((cOffset.l_c + i < LC)) {
        size_t val_c = get_global_range<Halo_Left, Cols>(cOffset.g_c + i);
        for (size_t j = 0; j < LR; j += cOffset.rLRng) {
          size_t val_r = get_global_range<Halo_Top, Rows>(cOffset.g_r + j);
          if ((cOffset.l_r + j < LR)) {
            tools::tuple::get<Index>(t)
                .get_pointer()[(cOffset.l_c + i) + (LC * (cOffset.l_r + j))] =
                tools::convert<typename MemoryTrait<
                    Memory_Type, decltype(tools::tuple::get<Index>(t))>::Type>(
                    tools::tuple::get<N>(t).get_pointer()[calculate_index(
                        val_c, val_r, Cols, Rows)]);
            /// FIXME: image cannot be accessed by pointer
          }
        }
      }
    }
    // here you need to put local barrier
    cOffset.barrier();
  }
};

// template deduction for Fill struct.
template <size_t Halo_Top, size_t Halo_Left, size_t Halo_Butt,
          size_t Halo_Right, size_t Offset, size_t LC, size_t LR, typename Expr,
          typename Loc, typename... Params>
static void fill_local_neighbour(Loc &cOffset,
                                 const tools::tuple::Tuple<Params...> &t) {
  Fill<Expr, Loc, Params...>::template fill_neighbour<
      Halo_Top, Halo_Left, Halo_Butt, Halo_Right, Offset, LC, LR>(cOffset, t);
}
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_LOAD_PATTERN_SQUARE_PATTERN_HPP_

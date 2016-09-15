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

/// \file expr_tree.hpp
/// \brief This file contains the forward declarations and all the necessary
/// include headers for the construction of static expression trees.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_EXPR_TREE_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_EXPR_TREE_HPP_

namespace visioncpp {
namespace internal {
/// \struct GlobalUnaryOp
/// \brief This class is used to encapsulate the global unary functor and the
/// types of each operand in this functor. The functor passed to this struct
/// applies global neighbour. This struct is used for global neighbour operation
/// template parameters:
/// \tparam USROP : the user/built-in functor
/// \tparam InTp the input type for that unary functor
template <typename USROP, typename InTp>
struct GlobalUnaryOp {
  using OP = USROP;
  using InType = InTp;
  visioncpp::internal::GlobalNeighbour<InTp> x;
  using OutType = decltype(OP()(x));
  static constexpr size_t Operation_type =
      internal::ops_category::GlobalNeighbourOP;
};
/// \struct LocalUnaryOp
/// \brief This class is used to encapsulate the local unary functor and the
/// types of each operand in this functor. The functor passed to this struct
/// applies local neighbour. This struct is used for local neighbour operation
/// template parameters:
/// \tparam USROP : the user/built-in functor
/// \tparam InTp the input type for that unary functor
template <typename USROP, typename InTp>
struct LocalUnaryOp {
  using OP = USROP;
  using InType = InTp;
  visioncpp::internal::LocalNeighbour<InTp> x;
  using OutType = decltype(OP()(x));
  static constexpr size_t Operation_type = internal::ops_category::NeighbourOP;
};
/// \struct LocalBinaryOp
/// \brief This class is used to encapsulate the local binary functor and the
/// types of each operand in this functor. The functor passed to this struct
/// applies local neighbour. This struct is used for local neighbour operation.
/// template parameters:
/// \tparam USROP : the user/built-in functor
/// \tparam InTp1 the left hand side input type for the binary functor
/// \tparam InTp2 the right hand side input type for the binary functor
template <typename USROP, typename InTp1, typename InTp2>
struct LocalBinaryOp {
  using OP = USROP;
  using InType1 = InTp1;
  using InType2 = InTp2;
  visioncpp::internal::LocalNeighbour<InTp1> x;
  visioncpp::internal::ConstNeighbour<InTp2> y;
  using OutType = decltype(OP()(x, y));
  static constexpr size_t Operation_type = internal::ops_category::NeighbourOP;
};
/// \struct PixelUnaryOp
/// \brief This class is used to encapsulate the unary point operation functor
/// and the types of each operand in this functor. The functor passed to this
/// struct applies point operation. This struct is used for point operation.
/// template parameters:
/// \tparam USROP : the user/built-in functor
/// \tparam InTp the right-hand side input type for the unary functor
template <typename USROP, typename InTp>
struct PixelUnaryOp {
  using OP = USROP;
  using InType = InTp;
  InType x;
  using OutType = decltype(OP()(x));
};
/// \struct PixelBinaryOp
/// \brief This class is used to encapsulate the binary point operation functor
/// and the types of each operand in this functor. The functor passed to this
/// struct applies point operation. This struct is used for point operation.
/// template parameters:
/// \tparam USROP : the user/built-in functor
/// \tparam InTp1 the left-hand side input type for the binary functor
/// \tparam InTp2 the right-hand side input type for the binary functor
template <typename USROP, typename InTp1, typename InTp2>
struct PixelBinaryOp {
  using OP = USROP;
  using InType1 = InTp1;
  using InType2 = InTp2;
  InType1 x;
  InType2 y;
  using OutType = decltype(OP()(x, y));
};

}  // internal
}  // visioncpp
// Expression Tree headers
#include "neighbour_ops/neighbour_ops.hpp"
#include "point_ops/point_ops.hpp"
// here it should be
#include "complex_ops/complex_ops.hpp"
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_EXPR_TREE_HPP_

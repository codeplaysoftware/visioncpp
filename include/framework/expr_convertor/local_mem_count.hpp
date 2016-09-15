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

/// \file local_mem_count.hpp
/// \brief LocalmemCount used to counting terminal nodes. The total number of
/// leaf nodes is used by MakePlaceHolderExprHelper to statically find the order
/// of the leaf node in a expression tree.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_LOCAL_MEM_COUNT_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_LOCAL_MEM_COUNT_HPP_

namespace visioncpp {
namespace internal {
/// \brief Partial specialisation of the LocalmemCount when the LeafNode contains
/// the const variable. In this case there is no shared memory for const
/// variable and the const variable is directly copied to the device and accessed
/// by each thread.
template <size_t N, size_t Rows, size_t Cols, size_t LVL>
struct LocalMemCount<
    expr_category::Unary,
    LeafNode<PlaceHolder<memory_type::Const, N, Cols, Rows, scope::Global>,
             LVL>> {
  static constexpr size_t Count = 0;
};
/// \brief Partial specialisation of the LocalmemCount when the LeafNode contains
/// a sycl buffer created on constant memory. There is no shared memory created
/// for a buffer on a device constant memory. Such a buffer is directly accessed
/// on the device by each thread.
template <size_t MemoryType, size_t N, size_t Rows, size_t Cols, size_t LVL>
struct LocalMemCount<
    expr_category::Unary,
    LeafNode<PlaceHolder<MemoryType, N, Cols, Rows, scope::Constant>, LVL>> {
  static constexpr size_t Count = 0;
};

/// \brief specialisation of LocalmemCount when the node is a LeafNode
/// template parameters:
/// \param RHS is the visionMemory
/// \param LVL shows the level of the node in the expression tree
template <typename RHSExpr, size_t LVL>
struct LocalMemCount<expr_category::Unary, LeafNode<RHSExpr, LVL>> {
  static constexpr size_t Count = 1;
};

/// template parameters:
/// \brief specialisation of LocalmemCount when the node has one child
/// \param RHS is the right-hand side expression of the node
/// \param LVL shows the level of the node in the expression tree
template <typename Expr>
struct LocalMemCount<expr_category::Unary, Expr> {
  static constexpr size_t Count =
      1 +
      LocalMemCount<Expr::RHSExpr::ND_Category, typename Expr::RHSExpr>::Count;
};
/// \brief specialisation of LocalmemCount when the node has two children
/// template parameters:
/// \param LHS is the left-hand side expression in the node
/// \param RHS is the right-hand side expression in the node
/// \param LVL shows the level of the node in the expression tree
template <typename Expr>
struct LocalMemCount<expr_category::Binary, Expr> {
  static constexpr size_t Count =
      1 +
      LocalMemCount<Expr::LHSExpr::ND_Category, typename Expr::LHSExpr>::Count +
      LocalMemCount<Expr::RHSExpr::ND_Category, typename Expr::RHSExpr>::Count;
};

/// \brief Specialisation of ExtractAccessor class where the expression node is
/// Assign
template <typename LHSExpr, typename RHSExpr, size_t Cols, size_t Rows,
          size_t LeafType, size_t LVL>
struct LocalMemCount<expr_category::Binary,
                     Assign<LHSExpr, RHSExpr, Cols, Rows, LeafType, LVL>> {
  static constexpr size_t Count =
      0 + LocalMemCount<RHSExpr::ND_Category, RHSExpr>::Count;
};

/// \brief Specialisation of ExtractAccessor class where the expression node is
/// is ParallelCopy (partial assign)
template <typename LHSExpr, typename RHSExpr, size_t Cols, size_t Rows,
          size_t OffsetColIn, size_t OffsetRowIn, size_t OffsetColOut,
          size_t OffsetRowOut, size_t LeafType, size_t LVL>
struct LocalMemCount<
    expr_category::Binary,
    ParallelCopy<LHSExpr, RHSExpr, Cols, Rows, OffsetColIn, OffsetRowIn,
                 OffsetColOut, OffsetRowOut, LeafType, LVL>> {
  static constexpr size_t Count =
      0 + LocalMemCount<RHSExpr::ND_Category, RHSExpr>::Count;
};

}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_LOCAL_MEM_COUNT_HPP_

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

/// \file leaf_count.hpp
/// \brief LeafCount used to counting terminal nodes. The total number of
/// leaf nodes is used by MakePlaceHolderExprHelper to statically find the order
/// of the leaf node in a expression tree.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_LEAF_COUNT_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_LEAF_COUNT_HPP_

namespace visioncpp {
namespace internal {
/// \brief specialisation of LeafCount when the node is a LeafNode
/// template parameters:
/// \param RHS is the visionMemory
/// \param LVL shows the level of the node in the expression tree
template <typename RHS, size_t LVL>
struct LeafCount<expr_category::Unary, LeafNode<RHS, LVL>> {
  static constexpr size_t Count = 1;
};

/// \brief specialisation of LeafCount when the node has one child
/// template parameters:
/// \param RHS is the right-hand side expression of the node
/// \param LVL shows the level of the node in the expression tree
template <typename Expr>
struct LeafCount<expr_category::Unary, Expr> {
  static constexpr size_t Count =
      0 + LeafCount<Expr::RHSExpr::ND_Category, typename Expr::RHSExpr>::Count;
};
/// \brief specialisation of LeafCount when the node has two children
/// template parameters:
/// \param LHS is the left-hand side expression in the node
/// \param RHS is the right-hand side expression in the node
/// \param LVL shows the level of the node in the expression tree
template <typename Expr>
struct LeafCount<expr_category::Binary, Expr> {
  static constexpr size_t Count =
      0 + LeafCount<Expr::LHSExpr::ND_Category, typename Expr::LHSExpr>::Count +
      LeafCount<Expr::RHSExpr::ND_Category, typename Expr::RHSExpr>::Count;
};
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_LEAF_COUNT_HPP_
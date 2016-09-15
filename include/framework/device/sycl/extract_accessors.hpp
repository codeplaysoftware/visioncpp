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

/// \file extract_accessors.hpp
/// \brief This files is used to provide an access mechanism for terminal nodes
/// on the device.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_EXTRACT_ACCESSORS_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_EXTRACT_ACCESSORS_HPP_

namespace visioncpp {
namespace internal {
///
/// \brief The extract accessor struct is used to extract the accessor from the
/// leafnodes and pack them in a tuple by using the in-order traverse algorithm
/// on the expression tree. As a different non-terminal node has different
/// behaviour the specialisation of the accessor extraction struct is required
/// per node.
///
template <size_t Category, typename Expr>
struct ExtractAccessor;

/// \brief Specialisation of ExtractAccessor class where the expression node is
/// LeafNode.
template <typename RHS, size_t LVL>
struct ExtractAccessor<expr_category::Unary, LeafNode<RHS, LVL>> {
  /// getting read access when the leaf node is accessed by traversing the right-
  /// hand side of an Assign or ParallelCopy (partial Assign) expression
  static tools::tuple::Tuple<
      typename RHS::template Accessor<cl::sycl::access::mode::read>>
  getTuple(cl::sycl::handler &cgh, LeafNode<RHS, LVL> &expr) {
    return expr.vilibMemory
        .template get_device_accessor<cl::sycl::access::mode::read>(cgh);
  }
  /// getting write access when the leaf node is accessed by traversing the left-
  /// hand side of ParallelCopy(partial assign) expression
  static tools::tuple::Tuple<
      typename RHS::template Accessor<cl::sycl::access::mode::write>>
  getWriteTuple(cl::sycl::handler &cgh, LeafNode<RHS, LVL> &expr) {
    return expr.vilibMemory
        .template get_device_accessor<cl::sycl::access::mode::write>(cgh);
  }
  /// getting discard_write access when the leaf node is accessed by traversing
  /// the left-hand side of an Assign expression
  static tools::tuple::Tuple<
      typename RHS::template Accessor<cl::sycl::access::mode::discard_write>>
  getDiscardWriteTuple(cl::sycl::handler &cgh, LeafNode<RHS, LVL> &expr) {
    return expr.vilibMemory
        .template get_device_accessor<cl::sycl::access::mode::discard_write>(
            cgh);
  }
};

/// \brief Specialisation of ExtractAccessor class where the expression node
/// has one child
template <typename Expr>
struct ExtractAccessor<expr_category::Unary, Expr> {
  static auto getTuple(cl::sycl::handler &cgh, Expr &expr)
      -> decltype(ExtractAccessor<Expr::RHSExpr::ND_Category,
                                  typename Expr::RHSExpr>::getTuple(cgh,
                                                                    expr.rhs)) {
    auto RHSTuple =
        ExtractAccessor<Expr::RHSExpr::ND_Category,
                        typename Expr::RHSExpr>::getTuple(cgh, expr.rhs);
    return RHSTuple;
  }
};

/// \brief Specialisation of ExtractAccessor class where the expression node
/// has two children
template <typename Expr>
struct ExtractAccessor<expr_category::Binary, Expr> {
  static auto getTuple(cl::sycl::handler &cgh, Expr &expr)
      -> decltype(tools::tuple::append(
          ExtractAccessor<Expr::LHSExpr::ND_Category,
                          typename Expr::LHSExpr>::getTuple(cgh, expr.lhs),
          ExtractAccessor<Expr::RHSExpr::ND_Category,
                          typename Expr::RHSExpr>::getTuple(cgh, expr.rhs))) {
    auto LHSTuple =
        ExtractAccessor<Expr::LHSExpr::ND_Category,
                        typename Expr::LHSExpr>::getTuple(cgh, expr.lhs);
    auto RHSTuple =
        ExtractAccessor<Expr::RHSExpr::ND_Category,
                        typename Expr::RHSExpr>::getTuple(cgh, expr.rhs);
    return tools::tuple::append(LHSTuple, RHSTuple);
  }
};

/// \brief Specialisation of ExtractAccessor class where the expression node is
/// Assign
template <typename LHSExpr, typename RHSExpr, size_t Cols, size_t Rows,
          size_t LeafType, size_t LVL>
struct ExtractAccessor<expr_category::Binary,
                       Assign<LHSExpr, RHSExpr, Cols, Rows, LeafType, LVL>> {
  static auto getTuple(
      cl::sycl::handler &cgh,
      Assign<LHSExpr, RHSExpr, Cols, Rows, LeafType, LVL> &expr)
      -> decltype(tools::tuple::append(
          ExtractAccessor<LHSExpr::ND_Category, LHSExpr>::getDiscardWriteTuple(
              cgh, expr.lhs),
          ExtractAccessor<RHSExpr::ND_Category, RHSExpr>::getTuple(cgh,
                                                                   expr.rhs))) {
    auto LHSTuple =
        ExtractAccessor<LHSExpr::ND_Category, LHSExpr>::getDiscardWriteTuple(
            cgh, expr.lhs);
    auto RHSTuple =
        ExtractAccessor<RHSExpr::ND_Category, RHSExpr>::getTuple(cgh, expr.rhs);
    return tools::tuple::append(LHSTuple, RHSTuple);
  }
};

/// \brief Specialisation of ExtractAccessor class where the expression node is
/// is a ParallelCopy (partial assign)
template <typename LHSExpr, typename RHSExpr, size_t Cols, size_t Rows,
          size_t OffsetColIn, size_t OffsetRowIn, size_t OffsetColOut,
          size_t OffsetRowOut, size_t LeafType, size_t LVL>
struct ExtractAccessor<
    expr_category::Binary,
    ParallelCopy<LHSExpr, RHSExpr, Cols, Rows, OffsetColIn, OffsetRowIn,
                 OffsetColOut, OffsetRowOut, LeafType, LVL>> {
  static auto getTuple(
      cl::sycl::handler &cgh,
      ParallelCopy<LHSExpr, RHSExpr, Cols, Rows, OffsetColIn, OffsetRowIn,
                   OffsetColOut, OffsetRowOut, LeafType, LVL> &expr)
      -> decltype(tools::tuple::append(
          ExtractAccessor<LHSExpr::ND_Category, LHSExpr>::getWriteTuple(
              cgh, expr.lhs),
          ExtractAccessor<RHSExpr::ND_Category, RHSExpr>::getTuple(cgh,
                                                                   expr.rhs))) {
    auto LHSTuple =
        ExtractAccessor<LHSExpr::ND_Category, LHSExpr>::getWriteTuple(cgh,
                                                                      expr.lhs);
    auto RHSTuple =
        ExtractAccessor<RHSExpr::ND_Category, RHSExpr>::getTuple(cgh, expr.rhs);
    return tools::tuple::append(LHSTuple, RHSTuple);
  }
};
}  // internal

template <typename Expr>
auto extract_accessors(cl::sycl::handler &cgh, Expr e) -> decltype(
    internal::ExtractAccessor<Expr::ND_Category, Expr>::getTuple(cgh, e)) {
  return internal::ExtractAccessor<Expr::ND_Category, Expr>::getTuple(cgh, e);
}

}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_EXTRACT_ACCESSORS_HPP_
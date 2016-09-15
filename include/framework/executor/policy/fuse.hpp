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

/// \file fuse.hpp
/// \brief This file contains the specialisation of the FuseExpr for terminal and
/// non-terminal nodes.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXECUTOR_POLICY_FUSE_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXECUTOR_POLICY_FUSE_HPP_

namespace visioncpp {
namespace internal {
/// \brief the FuseExpr when the expression type is not a terminal node
/// (leafNode).
template <size_t LCIn, size_t LRIn, size_t LCT, size_t LRT, typename Expr>
struct FuseExpr {
  /// \brief the fuse function for executing the given expr.
  /// \param expr : the expression passed to be executed on the device
  /// \param dev : the selected device for executing the expression
  /// return void
  template <typename DeviceT>
  static void fuse(Expr &expr, const DeviceT &dev) {
    /// LRT is the  workgroup size row and is checked with LR. LR is based
    /// on LRIn. LCT is the workgroup size column checked with LC. LC is based
    /// on LCIn. The local memory size  and the work group size is calculated at
    /// compile time.
    constexpr size_t LR =
        tools::IfConst<(Expr::RThread > LRIn), LRIn, Expr::RThread>::Value;
    constexpr size_t LC =
        tools::IfConst<(Expr::CThread > LCIn), LCIn, Expr::CThread>::Value;

    constexpr int rLThread =
        tools::IfConst<(Expr::RThread > LRT), LRT, Expr::RThread>::Value;

    constexpr int cLThread =
        tools::IfConst<(Expr::CThread > LCT), LCT, Expr::CThread>::Value;

    constexpr size_t rGThreads =
        (tools::IfConst<(Expr::RThread % LR == 0), (Expr::RThread / LR),
                        ((Expr::RThread / LR) + 1)>::Value) *
        rLThread;

    constexpr size_t cGThreads =
        (tools::IfConst<(Expr::CThread % LC == 0), (Expr::CThread / LC),
                        ((Expr::CThread / LC) + 1)>::Value) *
        cLThread;
    dev.template execute<LC, LR, cGThreads, rGThreads, cLThread, rLThread>(
        expr);
  }
};
/// \brief specialisation of Fuse struct when the Expr is a terminal node
/// (leafNode)
template <size_t LC, size_t LR, size_t LCT, size_t LRT, size_t LVL,
          typename RHS>
struct FuseExpr<LC, LR, LCT, LRT, LeafNode<RHS, LVL>> {
  /// when the node is a terminal node (leafNode) we do nothing as there is no
  /// need to run any expression
  template <typename DeviceT>
  static void fuse(LeafNode<RHS, LVL> &expr, const DeviceT &dev) {}
};

/// function fuse
/// \brief fuse function to generate a device kernel and execute.
/// template parameters:
/// \tparam LC: suggested column size for the local memory
/// \tparam LR: suggested row size for the local memory
/// \tparam LRT: suggested workgroup row size
/// \tparam LCT: suggested workgroup column size
/// \tparam Expr: the expression type
/// function parameters:
/// \param expr: the input expression
/// \param dev : the selected device for executing the expression
template <size_t LC, size_t LR, size_t LCT, size_t LRT, typename Expr,
          typename DeviceT>
inline void fuse(Expr expr, const DeviceT &dev) {
  FuseExpr<LC, LR, LCT, LRT, Expr>::fuse(expr, dev);
};
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXECUTOR_POLICY_FUSE_HPP_

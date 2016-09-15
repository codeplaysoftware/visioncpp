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

/// \file forward_declarations.hpp
/// \brief forward declarations for the library

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_FORWARD_DECLARATIONS_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_FORWARD_DECLARATIONS_HPP_

/// \brief VisionCpp namespace
namespace visioncpp {
/// \brief Scope is used to define the scope of the memory on the device
namespace scope {
using ScopeType = size_t;
static constexpr ScopeType Global = 0;
static constexpr ScopeType Local = 1;
static constexpr ScopeType Constant = 2;
static constexpr ScopeType Host_Buffer = 3;
}

/// \brief defines terminal nodes that can be used
namespace memory_type {
static constexpr size_t Host = 0;
static constexpr size_t Buffer1D = 1;
static constexpr size_t Buffer2D = 2;
static constexpr size_t Image = 3;
static constexpr size_t Const = 4;
}

/// \brief defines Executor policies available
namespace policy {
constexpr static bool Fuse = true;
constexpr static bool NoFuse = false;
}

/// \class backend
/// \brief enum class that defines supported backends.
enum class backend {
  /// represents sycl backend.
  sycl,
  /// number of backends.
  size
};

/// \class device
/// \brief enum class that defines supported devices types.
enum class device {
  /// represents the cpu device.
  cpu,
  /// represents the gpu device.
  gpu,
  /// represents the host device.
  host,
  /// number of devices
  size
};

/// \brief Internal implementations. Items from this scope should not be exposed
/// to the end user.
namespace internal {
/// \class Device_
/// class used to implement the execution of the expression tree.
/// \tparam BK is used to determine the backend
/// \tparam  DV is used to determine the selected device for that backend
template <backend BK, device DV>
class Device_;
}

/// \brief template deduction function for Device_ class
/// \tparam BK is used to determine the backend
/// \tparam  DV is used to determine the selected device for that backend
/// \return Device_
template <backend BK, device DV>
internal::Device_<BK, DV> make_device() {
  return internal::Device_<BK, DV>();
}

template <bool ExecPolicy, typename Expr, typename DeviceT>
void execute(Expr &, DeviceT &);

template <bool ExecPolicy, size_t LC, size_t LR, size_t LCT, size_t LRT,
          typename Expr, typename DeviceT>
void execute(Expr &, const DeviceT &);

namespace internal {
/// \brief this is used to define the type of nodes in an expression tree
namespace expr_category {
static constexpr size_t Nullary = 0;
static constexpr size_t Unary = 1;
static constexpr size_t Binary = 2;
};

/// \brief list of supported types of operations
/// the operation type can be Point operation, neighbour operation
namespace ops_category {
constexpr static size_t PointOP = 0;
constexpr static size_t NeighbourOP = 1;
constexpr static size_t GlobalNeighbourOP = 2;
};

/// \brief the definition is in \ref VirtualMemory
template <bool PlcType, typename Node, size_t LC = 8, size_t LR = 8,
          size_t LCT = 8, size_t LRT = 8>
struct VirtualMemory;

/// \brief the definition is in \ref LeafNode.
template <typename RHS, size_t LVL>
struct LeafNode;

/// \brief The definition is in \ref RUnOP file.
template <typename UN_OP, typename RHS, size_t Cols, size_t Rows, size_t LfType,
          size_t LVL>
struct RUnOP;

/// \brief The definition is in \ref RBiOP file.
template <typename BI_OP, typename LHS, typename RHS, size_t Cols, size_t Rows,
          size_t LfType, size_t LVL>
struct RBiOP;

/// \brief The definition is in \ref StnFilt file.
template <typename FilterOP, size_t Halo_T, size_t Halo_L, size_t Halo_B,
          size_t Halo_R, typename LHS, typename RHS, size_t Cols, size_t Rows,
          size_t LfType, size_t LVL>
struct StnFilt;

/// \brief The definition is in \ref StnNoFilt file.
template <typename FilterOP, size_t Halo_T, size_t Halo_L, size_t Halo_B,
          size_t Halo_R, typename RHS, size_t Cols, size_t Rows, size_t LfType,
          size_t LVL>
struct StnNoFilt;

/// \brief The definition is in \ref RDCN file.
template <typename DownSmplOP, typename RHS, size_t Cols, size_t Rows,
          size_t LfType, size_t LVL>
struct RDCN;
/// \brief The definition is in \ref ParallelCopy file.
template <typename LHS, typename RHS, size_t Cols, size_t Rows,
          size_t OffsetColIn, size_t OffsetRowIn, size_t OffsetColOut,
          size_t OffsetRowOut, size_t LfType, size_t LVL>
struct ParallelCopy;

/// \brief The definition is in \ref Assign file.
template <typename LHS, typename RHS, size_t Cols, size_t Rows, size_t LfType,
          size_t LVL>
struct Assign;

/// \brief The definition is in \ref SubExprRes file.
template <size_t LC, size_t LR, size_t LCT, size_t LRT, size_t LVL,
          typename Expr>
struct SubExprRes;

template <size_t LC, size_t LR, size_t LCT, size_t LRT, typename Expr,
          typename DeviceT>
void fuse(Expr, const DeviceT &);

template <size_t LC, size_t LR, size_t LCT, size_t LRT, typename Expr,
          typename DeviceT>
void no_fuse(Expr, const DeviceT &);

template <bool Conds, typename Expr, size_t LC, size_t LR, size_t LRT,
          size_t LCT, typename NestedExpr, typename DeviceT>
inline typename Expr::Sub_expression_Type execute_expr(NestedExpr,
                                                       const DeviceT &);

template <bool Conds, typename Expr, size_t LC, size_t LR, size_t LRT,
          size_t LCT, typename LHSExpr, typename RHSExpr, typename DeviceT>
inline typename Expr::Sub_expression_Type execute_expr(LHSExpr, RHSExpr,
                                                       const DeviceT &);
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_FORWARD_DECLARATIONS_HPP_

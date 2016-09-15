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

/// \file complex_ops.hpp
/// \brief This file contains a set of includes and forward declaration required
/// to build complex operations like pyramid.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_COMPLEX_OPS_COMPLEX_OPS_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_COMPLEX_OPS_COMPLEX_OPS_HPP_

namespace visioncpp {
namespace internal {
/// \brief CreatePyramidTupleType: This file is used to create each output
/// element type for each downsampling output of the pyramid memory. It is used
/// when we need the pyramid node auto generate the output node.
/// template parameters:
/// \tparam SatisfyingConds: a boolean variable is used to determine the end of
/// the recursive creation of the pyramid output tuple.
/// \tparam Cols: determines the column size of the input  pyramid
/// \tparam Rows: determines the row size of the input pyramid
/// \tparam LeafType: determines the type of the leafNode {Buffer2D, Buffer1D,
/// Host, Image}
/// \tparam Depth: represents the depth of down sampling
/// \tparam CurrentDepth: represents the number of output created so far in the
/// recursion
/// \tparam LHS is the final output of the pyramid combining all the node
/// together
/// \tparam ChildType... : is the total number of output type generated so far
template <bool SatisfyingConds, size_t Cols, size_t Rows, size_t LeafType,
          size_t Depth, size_t CurrentDepth, typename LHS,
          typename... ChildType>
struct CreatePyramidTupleType;

/// \brief CreatePyramidTuple. Once the type of each output element for the
/// output element of the pyramid has been created, The CreatePyramidTuple is
/// used to instantiate the output elements of the Tuple.
/// \tparam SatisfyingConds: a boolean variable is used to determine the end of
/// the recursive creation of the pyramid output tuple.
/// \tparam Cols: determines the column size of the input  pyramid
/// \tparam Rows: determines the row size of the input pyramid
/// \tparam LeafType: determines the type of the leafNode {Buffer2D, Buffer1D,
/// Host, Image}
/// \tparam Depth: represents the depth of down sampling
/// \tparam CurrentDepth: represents the number of outputs created so far in the
/// recursion
/// \tparam LHS is the final output of the pyramid combining all the nodes
/// together
template <bool SatisfyingConds, size_t Cols, size_t Rows, size_t LeafType,
          size_t Depth, size_t Current_Depth, typename LHS>
struct CreatePyramidTuple;

/// \brief create_pyramid_memory template deduction function for
/// CreatePyramidTuple struct.
/// template parameters:
/// the recursive creation of the pyramid output tuple.
/// \tparam Cols: determines the column size of the input  pyramid
/// \tparam Rows: determines the row size of the input pyramid
/// \tparam LeafType: determines the type of the leafNode {Buffer2D, Buffer1D,
/// Host, Image}
/// \tparam Depth: represents the depth of down sampling
/// \tparam CurrentDepth: represents the number of outputs created so far in the
/// recursion
/// \tparam LHS is the final output of the pyramid combining all the node
/// together.
/// function parameters:
/// \return CreatePyramidTupleType
template <size_t Cols, size_t Rows, size_t Depth, size_t Current_Depth,
          typename LHS>
typename CreatePyramidTupleType<false, Cols, Rows, LHS::Type::LeafType, Depth,
                                0, LHS>::Type inline create_pyramid_memory();

template <typename PyramidT, size_t N>
struct PyramidLeafNode {
  static constexpr bool has_out = false;
  using OutType = typename PyramidT::OutType;
  using Type = typename PyramidT::Type;
  using LHSExpr = typename PyramidT::LHSExpr;
  static constexpr size_t Level = PyramidT::Level;
  static constexpr size_t LeafType = PyramidT::LeafType;
  static constexpr bool SubExpressionEvaluationNeeded = true;
  static constexpr size_t Operation_type = PyramidT::Operation_type;
  static constexpr size_t RThread = PyramidT::RThread;
  static constexpr size_t CThread = PyramidT::CThread;
  static constexpr size_t ND_Category = PyramidT::ND_Category;
  static constexpr size_t Depth = PyramidT::Depth;
  using PyramidMem = typename PyramidT::PyramidMem;
  bool subexpr_execution_reseter;
  // pyramid
  PyramidT& rhs;  // the pyramid
  PyramidLeafNode(PyramidT& pst) : subexpr_execution_reseter(false), rhs(pst) {}

  void reset(bool reset) {
    rhs.subexpr_execution_reseter = reset;
    subexpr_execution_reseter = reset;
  }
  template <bool ForcedToExec, size_t LC, size_t LR, size_t LCT, size_t LRT,
            typename DeviceT>
  auto sub_expression_evaluation(const DeviceT& dev)
      -> decltype(tools::tuple::get<N>(rhs.mem)) {
    if (rhs.subexpr_execution_reseter) {
      if (rhs.first_time) {
        rhs.template sub_expression_evaluation<ForcedToExec, LC, LR, LCT, LRT>(
            dev);
        rhs.subexpr_execution_reseter = false;
        rhs.first_time = false;
        rhs.node_reseter = N;
      } else {
        if (N == rhs.node_reseter) {
          rhs.template sub_expression_evaluation<ForcedToExec, LC, LR, LRT,
                                                 LCT>(dev);
        }
      }
    }
    return tools::tuple::get<N>(rhs.mem);
  }
};
}  // end internal
}  // end visioncpp
#include "pyramid_mem.hpp"
#include "pyramid_with_auto_mem_gen.hpp"
#include "pyramid_with_auto_mem_sep.hpp"
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_COMPLEX_OPS_COMPLEX_OPS_HPP_

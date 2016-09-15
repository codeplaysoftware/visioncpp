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

/// \file pyramid_mem.hpp
/// \brief This file is used to create a pyramid output memory when the auto
/// generate output memory is selected. Here a tuple of output memory will be
/// constructed. The size of the tuple is a template variable determined based
/// on the depth of the pyramid.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXPR_COMPLEX_OPS_TREE_PYRAMID_MEM_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXPR_COMPLEX_OPS_TREE_PYRAMID_MEM_HPP_

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
/// \tparam CurrentDepth: represents the number of outputs created so far in the
/// recursion
/// \tparam LHS is the final output of the pyramid combining all the node
/// together
/// \tparam ChildType... : is the total number of output types generated so far
template <bool SatisfyingConds, size_t Cols, size_t Rows, size_t LeafType,
          size_t Depth, size_t CurrentDepth, typename LHS,
          typename... ChildType>
struct CreatePyramidTupleType {
  using SubBuffer = LeafNode<
      VisionMemory<false, LHS::Type::MemoryCategory, LHS::Type::LeafType,
                   typename LHS::Type::Scalar, Cols, Rows,
                   typename LHS::Type::ElementType, LHS::Type::Channels,
                   LHS::Type::scope, CurrentDepth + 100>,
      CurrentDepth + 100>;
  using Type = typename CreatePyramidTupleType<
      (Depth == CurrentDepth + 1), Cols / 2, Rows / 2, LeafType, Depth,
      CurrentDepth + 1, LHS, ChildType..., SubBuffer>::Type;
};

/// \brief specialisation of the CreatePyramidTupleType when the
/// SatisfyingConds is true. It does nothing but representing the end of
/// recursive constructing output types in the Tuple
template <size_t Cols, size_t Rows, size_t LeafType, size_t Depth,
          size_t CurrentDepth, typename LHS, typename... ChildType>
struct CreatePyramidTupleType<true, Cols, Rows, LeafType, Depth, CurrentDepth,
                              LHS, ChildType...> {
  using Type = tools::tuple::Tuple<ChildType...>;
};

/// \brief CreatePyramidTuple. Once the type of each output element for the
/// output element of the pyramid has been created, The CreatePyramidTuple is
/// used to instantiate the output elements of the Tuple.
/// \tparam SatisfyingConds: a boolean variable is used to determine the end of
/// the recursive creation of the pyramid output tuple.
/// \tparam Cols: determines the column size of the input pyramid
/// \tparam Rows: determines the row size of the input pyramid
/// \tparam LeafType: determines the type of the leafNode {Buffer2D, Buffer1D,
/// Host, Image}
/// \tparam Depth: represents the depth of down sampling
/// \tparam CurrentDepth: represents the number of output created so far in the
/// recursion
/// \tparam LHS is the final output of the pyramid combining all the nodes
/// together
template <bool SatisfyingConds, size_t Cols, size_t Rows, size_t LeafType,
          size_t Depth, size_t CurrentDepth, typename LHS>
struct CreatePyramidTuple {
  using SubBufferNode = LeafNode<
      VisionMemory<false, LHS::Type::MemoryCategory, LHS::Type::LeafType,
                   typename LHS::Type::Scalar, Cols, Rows,
                   typename LHS::Type::ElementType, LHS::Type::Channels,
                   LHS::Type::scope, CurrentDepth + 100>,
      CurrentDepth + 100>;

  using SubBuffer = typename SyclMem<false, LHS::Type::LeafType, LHS::Type::Dim,
                                     typename LHS::Type::ElementType>::Type;

  /// create_tuple function:
  /// \brief This function is used to instantiate the output tuple.
  /// \return Tuple
  static auto create_tuple() -> decltype(tools::tuple::append(
      tools::tuple::make_tuple(
          SubBufferNode(SubBuffer(get_range<LHS::Type::Dim>(Rows, Cols)))),
      CreatePyramidTuple<(Depth == (CurrentDepth + 1)), Cols / 2, Rows / 2,
                         LeafType, Depth, CurrentDepth + 1,
                         LHS>::create_tuple())) {
    auto subBufferExpr = tools::tuple::make_tuple(
        SubBufferNode(SubBuffer(get_range<LHS::Type::Dim>(Rows, Cols))));

    auto nestedExpr = CreatePyramidTuple<(Depth == (CurrentDepth + 1)),
                                         Cols / 2, Rows / 2, LeafType, Depth,
                                         CurrentDepth + 1, LHS>::create_tuple();

    return tools::tuple::append(subBufferExpr, nestedExpr);
  }
};

/// \brief specialisation of the CreatePyramidTuple when the
/// SatisfyingConds is true. It does nothing but representing the end of
/// recursive instantiating the output types in the Tuple
template <size_t Cols, size_t Rows, size_t LeafType, size_t Depth,
          size_t CurrentDepth, typename LHS>
struct CreatePyramidTuple<true, Cols, Rows, LeafType, Depth, CurrentDepth,
                          LHS> {
  using SubBufferNode = LeafNode<
      VisionMemory<false, LHS::Type::MemoryCategory, LHS::Type::LeafType,
                   typename LHS::Type::Scalar, Cols, Rows,
                   typename LHS::Type::ElementType, LHS::Type::Channels,
                   LHS::Type::scope, CurrentDepth + 100>,
      CurrentDepth + 100>;

  using SubBuffer = typename SyclMem<false, LHS::Type::LeafType, LHS::Type::Dim,
                                     typename LHS::Type::ElementType>::Type;

  /// create_tuple function:
  /// \brief This function is used to instantiate the output tuple.
  /// \return Tuple
  static decltype(tools::tuple::make_tuple()) inline create_tuple() {
    return tools::tuple::make_tuple();
  }
};

// template deduction for pyramid tuple
template <size_t Cols, size_t Rows, size_t Depth, size_t CurrentDepth,
          typename LHS>
typename CreatePyramidTupleType<false, Cols, Rows, LHS::Type::LeafType, Depth,
                                0, LHS>::Type inline create_pyramid_memory() {
  return CreatePyramidTuple<(Depth == 0), Cols, Rows, LHS::Type::LeafType,
                            Depth, 0, LHS>::create_tuple();
}
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXPR_COMPLEX_OPS_TREE_PYRAMID_MEM_HPP_
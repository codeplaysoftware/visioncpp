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

/// \file local_output.hpp
/// \brief This file contains the different specialisations of the LocalOutput

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_LOCAL_OUTPUT_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_LOCAL_OUTPUT_HPP_

namespace visioncpp {
namespace internal {
/// \brief OutputAccessor struct is used to generate an accessor when the node
/// is not root. When the node is root no local accessor will be created.
/// Therefore we eliminate the extra local memory for root node.
template <size_t IsRoot, size_t LeafType, size_t LC, size_t LR,
          typename OutType>
struct OutputAccessor {
  static auto getTuple(cl::sycl::handler &cgh)
      -> decltype(tools::tuple::make_tuple()) {
    return tools::tuple::make_tuple();
  }
};
/// \brief specialisation of OutputAccessor when a node is not a root node.
/// Here we create on output memory for the node.
template <size_t LeafType, size_t LC, size_t LR, typename OutType>
struct OutputAccessor<false, LeafType, LC, LR, OutType> {
  using Accessor = cl::sycl::accessor<OutType, MemDimension<LeafType>::Dim,
                                      cl::sycl::access::mode::read_write,
                                      cl::sycl::access::target::local>;
  static tools::tuple::Tuple<Accessor> getTuple(cl::sycl::handler &cgh) {
    /// In get range the column is changed with row in order to set x as
    /// column and y as row
    return Accessor(get_range<MemDimension<LeafType>::Dim>(LR, LC), cgh);
  }
};

/// \brief LocalOutput accessor. The local output does nothing when the
/// operation type is point operation. If it is called when the whole expression
/// tree is point op or global op no local memory will be created. When the
/// combination of point op and NeighbourOP is used we consider the expression
/// tree as a NeighbourOP expression tree and generate a set of local memory.
/// template parameters:
/// \tparam PointOp: determines whether or not the overall expression tree is
/// pointOp
/// \tparam IsRoot: determines whether or not a node is a root node.
/// \tparam LC: determines the column size of the local memory
/// \tparam LR: determines the row size of the local memory
/// \tparam Expr: determines the type of the expression
template <bool PointOp, size_t IsRoot, size_t LC, size_t LR, typename Expr>
struct LocalOutput {
  static constexpr size_t Out_LC = LC;
  static constexpr size_t Out_LR = LR;
  /// \brief getTuple function is used to create and wrap local memory into a
  /// tuple
  /// \param cgh : sycl command group handler. 
  static inline decltype(tools::tuple::make_tuple()) getTuple(
      cl::sycl::handler &cgh) {
    return tools::tuple::make_tuple();
  }
};

/// \brief specialisation of the LocalOutput for leaf node when the vision
/// memory is a const variable. We separate this from the specialisation of the
/// leaf node for LocalOutput because we don't need to create the local memory for
/// the Constant variable.
template <size_t IsRoot, bool in, size_t LC, size_t LR, size_t Width,
          size_t Height, size_t element_category, size_t LVL, typename Sclr>
struct LocalOutput<
    false, IsRoot, LC, LR,
    LeafNode<VisionMemory<in, element_category, memory_type::Const, Sclr, Width,
                          Height, Sclr, 1, scope::Global, LVL>,
             LVL>> {
  static constexpr size_t Out_LC = LC;
  static constexpr size_t Out_LR = LR;
  static decltype(tools::tuple::make_tuple()) getTuple(cl::sycl::handler &cgh) {
    return tools::tuple::make_tuple();
  }
};

/// \brief specialisation of the LocalOutput for leaf node when the vision
/// memory is a constant buffer. We separate this from the specialisation of the
/// leaf node for LocalOutput because we don't need to create the local memory for
/// an input data on a Constant buffer.
template <size_t IsRoot, bool in, size_t LC, size_t LR, size_t element_category,
          size_t Memory_Type, typename Scalar, size_t Width, size_t Height,
          typename ElementTp, size_t Elements, size_t LVL>
struct LocalOutput<
    false, IsRoot, LC, LR,
    LeafNode<VisionMemory<in, element_category, Memory_Type, Scalar, Width,
                          Height, ElementTp, Elements, scope::Constant, LVL>,
             LVL>> {
  static constexpr size_t Out_LC = LC;
  static constexpr size_t Out_LR = LR;
  static decltype(tools::tuple::make_tuple()) getTuple(cl::sycl::handler &cgh) {
    return tools::tuple::make_tuple();
  }
};

/// \brief LocalOutput specialisation for leaf node it creates the local
/// accessor that contains the data of the leaf node from global memory. This
/// local memory then will be used by other non-terminal node as an input value.
template <size_t IsRoot, size_t LC, size_t LR, size_t LVL, typename RHS>
struct LocalOutput<false, IsRoot, LC, LR, LeafNode<RHS, LVL>> {
  static constexpr size_t Out_LC = LC;
  static constexpr size_t Out_LR = LR;
  using Accessor = cl::sycl::accessor<typename RHS::ElementType, RHS::Dim,
                                      cl::sycl::access::mode::read_write,
                                      cl::sycl::access::target::local>;
  static tools::tuple::Tuple<Accessor> getTuple(cl::sycl::handler &cgh) {
    return Accessor(get_range<RHS::Dim>(LR, LC), cgh);
  }
};

/// \brief LocalOutput specialisation for unary operation(RUnOP) it creates the
/// local accessor to store the output of unary operation which is going to be
/// used as an output for its parent.
template <size_t IsRoot, size_t LC, size_t LR, typename OP, typename RHSExpr,
          size_t Cols, size_t Rows, size_t LeafType, size_t LVL>
struct LocalOutput<false, IsRoot, LC, LR,
                   RUnOP<OP, RHSExpr, Cols, Rows, LeafType, LVL>> {
  static constexpr size_t Out_LC =
      LocalOutput<false, false, LC, LR, RHSExpr>::Out_LC;
  static constexpr size_t Out_LR =
      LocalOutput<false, false, LC, LR, RHSExpr>::Out_LR;
  static auto getTuple(cl::sycl::handler &cgh) -> decltype(tools::tuple::append(
      LocalOutput<false, false, LC, LR, RHSExpr>::getTuple(cgh),
      OutputAccessor<IsRoot, LeafType, Out_LC, Out_LR,
                     typename OP::OutType>::getTuple(cgh))) {
    auto OutTuple = OutputAccessor<IsRoot, LeafType, Out_LC, Out_LR,
                                   typename OP::OutType>::getTuple(cgh);
    auto RHSTuple = LocalOutput<false, false, LC, LR, RHSExpr>::getTuple(cgh);
    return tools::tuple::append(RHSTuple, OutTuple);
  }
};

/// \brief LocalOutput specialisation for binary operation(RBiOP) it creates the
/// local accessor to store the output of binary operation which is going to be
/// used as an output for its parent.
template <size_t IsRoot, size_t LC, size_t LR, typename OP, typename LHSExpr,
          typename RHSExpr, size_t Cols, size_t Rows, size_t LeafType,
          size_t LVL>
struct LocalOutput<false, IsRoot, LC, LR,
                   RBiOP<OP, LHSExpr, RHSExpr, Cols, Rows, LeafType, LVL>> {
  static constexpr size_t lhs =
      LocalOutput<false, false, LC, LR, LHSExpr>::Out_LC *
      LocalOutput<false, false, LC, LR, LHSExpr>::Out_LR;
  static constexpr size_t rhs =
      LocalOutput<false, false, LC, LR, RHSExpr>::Out_LC *
      LocalOutput<false, false, LC, LR, RHSExpr>::Out_LR;
  static constexpr bool res = lhs < rhs;
  using Type = typename tools::StaticIf<res, LHSExpr, RHSExpr>::Type;

  static constexpr size_t Out_LC =
      LocalOutput<false, false, LC, LR, Type>::Out_LC;
  static constexpr size_t Out_LR =
      LocalOutput<false, false, LC, LR, Type>::Out_LR;

  static auto getTuple(cl::sycl::handler &cgh) -> decltype(tools::tuple::append(
      tools::tuple::append(
          LocalOutput<false, false, LC, LR, LHSExpr>::getTuple(cgh),
          LocalOutput<false, false, LC, LR, RHSExpr>::getTuple(cgh)),
      OutputAccessor<IsRoot, LeafType, Out_LC, Out_LR,
                     typename OP::OutType>::getTuple(cgh))) {
    auto OutTuple = OutputAccessor<IsRoot, LeafType, Out_LC, Out_LR,
                                   typename OP::OutType>::getTuple(cgh);
    auto LHSTuple = LocalOutput<false, false, LC, LR, LHSExpr>::getTuple(cgh);
    auto RHSTuple = LocalOutput<false, false, LC, LR, RHSExpr>::getTuple(cgh);
    return tools::tuple::append(tools::tuple::append(LHSTuple, RHSTuple),
                                OutTuple);
  }
};
/// \brief LocalOutput specialisation for binary neighbour operation(StnFilt).
/// It creates the local accessor to store the output of neighbour operation
/// which is going to be used as an output for its parent.
template <size_t IsRoot, size_t LC, size_t LR, typename OP, size_t Halo_T,
          size_t Halo_L, size_t Halo_B, size_t Halo_R, typename LHSExpr,
          typename RHSExpr, size_t Cols, size_t Rows, size_t LeafType,
          size_t LVL>
struct LocalOutput<false, IsRoot, LC, LR,
                   StnFilt<OP, Halo_T, Halo_L, Halo_B, Halo_R, LHSExpr, RHSExpr,
                           Cols, Rows, LeafType, LVL>> {
  static constexpr size_t Halo_COL = Halo_L + Halo_R;
  static constexpr size_t Halo_ROW = Halo_T + Halo_B;
  static constexpr size_t Out_LC =
      LocalOutput<false, false, LC + Halo_COL, LR + Halo_ROW, LHSExpr>::Out_LC -
      Halo_COL;
  static constexpr size_t Out_LR =
      LocalOutput<false, false, LC + Halo_COL, LR + Halo_ROW, LHSExpr>::Out_LR -
      Halo_ROW;

  static auto getTuple(cl::sycl::handler &cgh) -> decltype(tools::tuple::append(
      tools::tuple::append(
          LocalOutput<false, false, LC + Halo_COL, LR + Halo_ROW,
                      LHSExpr>::getTuple(cgh),
          LocalOutput<false, false, LC, LR, RHSExpr>::getTuple(cgh)),
      OutputAccessor<IsRoot, LeafType, Out_LC, Out_LR,
                     typename OP::OutType>::getTuple(cgh))) {
    auto OutTuple = OutputAccessor<IsRoot, LeafType, Out_LC, Out_LR,
                                   typename OP::OutType>::getTuple(cgh);
    auto LHSTuple = LocalOutput<false, false, LC + Halo_COL, LR + Halo_ROW,
                                LHSExpr>::getTuple(cgh);

    auto RHSTuple = LocalOutput<false, false, LC, LR, RHSExpr>::getTuple(cgh);

    return tools::tuple::append(tools::tuple::append(LHSTuple, RHSTuple),
                                OutTuple);
  }
};

/// \brief LocalOutput specialisation for unary neighbour operation(StnNoFilt).
/// It creates the local accessor to store the output of neighbour operation
/// which is going to be used as an output for its parent.
template <size_t IsRoot, size_t LC, size_t LR, typename OP, size_t Halo_T,
          size_t Halo_L, size_t Halo_B, size_t Halo_R, typename RHSExpr,
          size_t Cols, size_t Rows, size_t LeafType, size_t LVL>
struct LocalOutput<false, IsRoot, LC, LR,
                   StnNoFilt<OP, Halo_T, Halo_L, Halo_B, Halo_R, RHSExpr, Cols,
                             Rows, LeafType, LVL>> {
  static constexpr size_t Halo_COL = Halo_L + Halo_R;
  static constexpr size_t Halo_ROW = Halo_T + Halo_B;
  static constexpr size_t Out_LC =
      LocalOutput<false, false, LC + Halo_COL, LR + Halo_ROW, RHSExpr>::Out_LC -
      Halo_COL;
  static constexpr size_t Out_LR =
      LocalOutput<false, false, LC + Halo_COL, LR + Halo_ROW, RHSExpr>::Out_LR -
      Halo_ROW;

  static auto getTuple(cl::sycl::handler &cgh) -> decltype(tools::tuple::append(
      LocalOutput<false, false, LC + Halo_COL, LR + Halo_ROW,
                  RHSExpr>::getTuple(cgh),
      OutputAccessor<IsRoot, LeafType, Out_LC, Out_LR,
                     typename OP::OutType>::getTuple(cgh))) {
    auto OutTuple = OutputAccessor<IsRoot, LeafType, Out_LC, Out_LR,
                                   typename OP::OutType>::getTuple(cgh);
    auto RHSTuple = LocalOutput<false, false, LC + Halo_COL, LR + Halo_ROW,
                                RHSExpr>::getTuple(cgh);

    return tools::tuple::append(RHSTuple, OutTuple);
  }
};

/// \brief LocalOutput specialisation for reduction neighbour operation(RDCN).
/// It creates the local accessor to store the output of the operation
/// which is going to be used as an output for its parent.
template <size_t IsRoot, size_t LC, size_t LR, typename OP, typename RHSExpr,
          size_t Cols, size_t Rows, size_t LeafType, size_t LVL>
struct LocalOutput<false, IsRoot, LC, LR,
                   RDCN<OP, RHSExpr, Cols, Rows, LeafType, LVL>> {
  // Here we don't use x and y thread because it has already been calculated
  // through Out_LC and Out_LR. When we come here it means that local sampler is
  // used not the global one
  static constexpr size_t LC_Ratio = RHSExpr::Type::Cols / Cols;
  static constexpr size_t LR_Ratio = RHSExpr::Type::Rows / Rows;
  static constexpr size_t Out_LC =
      LocalOutput<false, false, LC, LR, RHSExpr>::Out_LC / LC_Ratio;
  static constexpr size_t Out_LR =
      LocalOutput<false, false, LC, LR, RHSExpr>::Out_LR / LR_Ratio;
  static auto getTuple(cl::sycl::handler &cgh) -> decltype(tools::tuple::append(
      LocalOutput<false, false, LC, LR, RHSExpr>::getTuple(cgh),
      OutputAccessor<IsRoot, LeafType, Out_LC, Out_LR,
                     typename OP::OutType>::getTuple(cgh))) {
    auto OutTuple = OutputAccessor<IsRoot, LeafType, Out_LC, Out_LR,
                                   typename OP::OutType>::getTuple(cgh);
    auto RHSTuple = LocalOutput<false, false, LC, LR, RHSExpr>::getTuple(cgh);
    return tools::tuple::append(RHSTuple, OutTuple);
  }
};

/// \brief LocalOutput specialisation for point operation(ParallelCopy).
/// It creates the local accessor to store the output of the operation
/// which is going to be used as an output for its parent.
template <size_t IsRoot, size_t LC, size_t LR, typename LHSExpr,
          typename RHSExpr, size_t Cols, size_t Rows, size_t OffsetColIn,
          size_t OffsetRowIn, size_t OffsetColOut, size_t OffsetRowOut,
          size_t LeafType, size_t LVL>
struct LocalOutput<
    false, IsRoot, LC, LR,
    ParallelCopy<LHSExpr, RHSExpr, Cols, Rows, OffsetColIn, OffsetRowIn,
                 OffsetColOut, OffsetRowOut, LeafType, LVL>> {
  static auto getTuple(cl::sycl::handler &cgh)
      -> decltype(LocalOutput<false, false, LC, LR, RHSExpr>::getTuple(cgh)) {
    return LocalOutput<false, false, LC, LR, RHSExpr>::getTuple(cgh);
  }
};

/// \brief LocalOutput specialisation for point operation(Assign).
/// It creates the local accessor to store the output of the operation
/// which is going to be used as an output for its parent.
template <size_t IsRoot, size_t LC, size_t LR, typename LHSExpr,
          typename RHSExpr, size_t Cols, size_t Rows, size_t LeafType,
          size_t LVL>
struct LocalOutput<false, IsRoot, LC, LR,
                   Assign<LHSExpr, RHSExpr, Cols, Rows, LeafType, LVL>> {
  static auto getTuple(cl::sycl::handler &cgh)
      -> decltype(LocalOutput<false, IsRoot, LC, LR, RHSExpr>::getTuple(cgh)) {
    return LocalOutput<false, IsRoot, LC, LR, RHSExpr>::getTuple(cgh);
  }
};

/// \brief create_local_accessors is a deduction function for creating local
/// accessor.
/// parameters:
/// \param cgh: sycl command group handler
/// \return Tuple

template <size_t LC, size_t LR, typename Expr>
inline auto create_local_accessors(cl::sycl::handler &cgh)
    -> decltype(LocalOutput<Expr::Operation_type != ops_category::NeighbourOP,
                            true, LC, LR, Expr>::getTuple(cgh)) {
  return LocalOutput<Expr::Operation_type != ops_category::NeighbourOP, true,
                     LC, LR, Expr>::getTuple(cgh);
}
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXPR_CONVERTOR_LOCAL_OUTPUT_HPP_
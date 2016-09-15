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

/// \file pyramid_with_auto_mem_sep.hpp
/// \brief This file contains the construction of the pyramid node where a
/// user can pass separable filter2d functor and general downsampling functor
/// to execute. Here, the user does need to pass the output memory. The output
/// memory will be created based on the depth of pyramid. Also an output file
/// will be created which merge all the result of the pyramid image in one file.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_COMPLEX_OPS_PYRAMID_WITH_AUTO_MEM_SEP_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_COMPLEX_OPS_PYRAMID_WITH_AUTO_MEM_SEP_HPP_

namespace visioncpp {
namespace internal {

/// \struct PyramidExecuteAutoMemSep
/// \brief here we execute the pyramid; automatically construct the output tuple
/// based on the depth of the pyramid; and construct the kernels.
/// template parameters:
/// \tparam SatisfyingConds: a boolean variable is used to determine the end of
/// the recursive creation of the pyramid kernel at compile time.
/// \tparam Fltr2DOP: general Filter2D functor
/// \tparam DownSmplOP: downsampling function for pyramid
/// \tparam Cols: determines the column size of the input pyramid
/// \tparam Rows: determines the row size of the input pyramid
/// \tparam LeafType: determines the type of the leafNode {Buffer2D, Buffer1D,
/// Host, Image}
/// \tparam LVL: the current level of the node based on the previous creation
/// of the kernel. This level express the a new level for RHS. This level is an
/// artificial level. In each step we consider an expression needed to be
/// executed by a kernel as a subexpression of a big expression tree. The LVL
/// here represent the level of the root of each subexpression inside that big
/// artificial expression.
/// \tparam LC: is the column size of local memory required by Filter2D and
/// DownSmplOP
/// \tparam LR: is the row size of local memory required by Filter2D and
/// DownSmplOP
/// \tparam LCT: is the column size of workgroup
/// \tparam LRT: is the row size of workgroup
/// \tparam Depth: represents the depth of down sampling
/// \tparam CurrentDepth: represents the number of kernel created so far by
/// recursively calling of PyramidExecuteGen struct.
/// \tparam LHS is the final output of the pyramid combining all the node
/// together
/// \tparam RHS is the intermediate input passed for the current kernel to be
/// \tparam SepFilterRow is the separable filter for row
/// \tparam SepFilterCol is the separable filter for col
/// \tparam PyramidMem: is a tuple of pyramid output memory
template <bool SatisfyingConds, typename SepFltrColOP, typename SepFltrRowOP,
          typename DownSmplOP, size_t Cols, size_t Rows, size_t LeafType,
          size_t OffsetCol, size_t OffsetRow, size_t LVL, size_t LC, size_t LR,
          size_t LCT, size_t LRT, size_t Depth, size_t CurrentDepth,
          typename LHS, typename RHS, typename SepFilterCol,
          typename SepFilterRow, typename PyramidMem>
struct PyramidExecuteAutoMemSep {
  /// function sub_execute
  /// \brief is the function used to construct a subexpression tree
  /// corresponding to the CurrentDepth of the pyramid and launch the kernel
  /// \param rhs: is the intermediate input passed for the current kernel to
  /// be executed as a leafNode
  /// \param spFltrRow is the separable filter node for Row
  /// \param spFltrCol is the separable filter node for Col
  /// \param mem: is the tuple of pyramid output memory
  /// \param dev : the selected device for executing the expression
  /// \return void
  template <typename DeviceT>
  static void sub_execute(RHS &rhs, SepFilterCol &spFltrCol,
                          SepFilterRow &spFltrRow, PyramidMem &mem,
                          const DeviceT &dev) {
    auto a = StnFilt<LocalBinaryOp<SepFltrColOP, typename RHS::OutType,
                                   typename SepFilterCol::OutType>,
                     SepFilterCol::Type::Rows / 2, SepFilterCol::Type::Cols / 2,
                     SepFilterCol::Type::Rows / 2, SepFilterCol::Type::Cols / 2,
                     RHS, SepFilterCol, Cols, Rows, LeafType, 1 + RHS::Level>(
        rhs, spFltrCol);

    auto b = StnFilt<LocalBinaryOp<SepFltrRowOP, typename decltype(a)::OutType,
                                   typename SepFilterRow::OutType>,
                     SepFilterRow::Type::Rows / 2, SepFilterRow::Type::Cols / 2,
                     SepFilterRow::Type::Rows / 2, SepFilterRow::Type::Cols / 2,
                     decltype(a), SepFilterRow, Cols, Rows, LeafType,
                     1 + decltype(a)::Level>(a, spFltrRow);
    auto c =
        RDCN<LocalUnaryOp<DownSmplOP, typename decltype(b)::OutType>,
             decltype(b), Cols / 2, Rows / 2, LeafType, 1 + decltype(b)::Level>(
            b);
    using RHSType = typename tools::RemoveAll<decltype(
        tools::tuple::get<CurrentDepth>(mem))>::Type;
    auto d = Assign<RHSType, decltype(c), Cols / 2, Rows / 2, LeafType, LVL>(
        tools::tuple::get<CurrentDepth>(mem), c);

    fuse<LC, LR, LCT, LRT>(d, dev);

    PyramidExecuteAutoMemSep<
        (Depth == (CurrentDepth + 1)), SepFltrColOP, SepFltrRowOP, DownSmplOP,
        Cols / 2, Rows / 2, LeafType, OffsetCol, OffsetRow + Rows / 2,
        RHSType::Level + 1, LC, LR, LCT, LRT, Depth, CurrentDepth + 1, LHS,
        RHSType, SepFilterCol, SepFilterRow,
        PyramidMem>::sub_execute(tools::tuple::get<CurrentDepth>(mem),
                                 spFltrCol, spFltrRow, mem, dev);
  }
};

/// \brief specialisation of the PyramidExecuteAutoMemSep when the
/// SatisfyingConds is true. It does nothing but representing the end of
/// recursive constructing and launching of an expression tree
template <typename SepFltrColOP, typename SepFltrRowOP, typename DownSmplOP,
          size_t Cols, size_t Rows, size_t LeafType, size_t OffsetCol,
          size_t OffsetRow, size_t LVL, size_t LC, size_t LR, size_t LRT,
          size_t LCT, size_t Depth, size_t CurrentDepth, typename LHS,
          typename RHS, typename SepFilterCol, typename SepFilterRow,
          typename PyramidMem>
struct PyramidExecuteAutoMemSep<true, SepFltrColOP, SepFltrRowOP, DownSmplOP,
                                Cols, Rows, LeafType, OffsetCol, OffsetRow, LVL,
                                LC, LR, LCT, LRT, Depth, CurrentDepth, LHS, RHS,
                                SepFilterCol, SepFilterRow, PyramidMem> {
  template <typename DeviceT>
  static void sub_execute(RHS &, SepFilterCol &, SepFilterRow &, PyramidMem &,
                          const DeviceT &) {}
};

/// \struct PyramidAutomemSep
/// \brief PyramidAutomemSep is used to construct a pyramid node in the
/// expression tree with two separable Filter for row and column and general
/// DownSmplOP functors. It automatically generates the tuple of output based on
/// the depth.
/// template parameters:
/// \tparam SepFltrColOP: separable Filter functor for col
/// \tparam SepFltrRowOP: separable Filter functor for row
/// \tparam DownSmplOP: downsampling function for pyramid
/// \tparam LHS is the input passed for pyramid
/// \tparam SepFilterCol: is the separable filter node for column
/// \tparam SepFilterRow: is the separable filter node for row
/// \tparam Cols: determines the column size of the input pyramid
/// \tparam Rows: determines the row size of the input pyramid
/// \tparam LfType: determines the type of the leafNode {Buffer2D, Buffer1D,
/// Host, Image}
/// \tparam LVL: the level of the node in the expression tree
/// \tparam Dp: represents the depth of downsampling
template <typename SepFltrColOP, typename SepFltrRowOP, typename DownSmplOP,
          typename RHS, typename SepFilterCol, typename SepFilterRow,
          size_t Cols, size_t Rows, size_t LfType, size_t LVL, size_t Dp>
struct PyramidAutomemSep {
 public:
  static constexpr bool has_out = false;
  using OutType = typename DownSmplOP::OutType;
  using Type =
      typename OutputMemory<OutType, LfType, Cols + Cols / 2, Rows, LVL>::Type;
  using LHSExpr = LeafNode<Type, LVL>;
  static constexpr size_t Level = LVL;
  static constexpr size_t LeafType = Type::LeafType;
  static constexpr bool SubExpressionEvaluationNeeded = true;
  static constexpr size_t Operation_type = ops_category::GlobalNeighbourOP;
  static constexpr size_t RThread = Rows;
  static constexpr size_t CThread = Cols;
  static constexpr size_t ND_Category = expr_category::Unary;
  static constexpr size_t Depth = Dp;
  using PyramidMem =
      typename CreatePyramidTupleType<false, Cols / 2, Rows / 2, LeafType,
                                      Depth, 0, LHSExpr>::Type;
  RHS rhs;
  SepFilterCol spFltrCol;
  SepFilterRow spFltrRow;

  bool subexpr_execution_reseter;
  bool first_time;
  size_t node_reseter;
  PyramidMem mem;
  PyramidAutomemSep(RHS rhsArg, SepFilterCol fltrCol, SepFilterRow fltrRow)
      : rhs(rhsArg),
        spFltrCol(fltrCol),
        spFltrRow(fltrRow),
        subexpr_execution_reseter(false),
        first_time(true),
        node_reseter(0),
        mem(create_pyramid_memory<Cols / 2, Rows / 2, Depth, 0, LHSExpr>()) {}

  void reset(bool reset) {
    rhs.reset(reset);
    subexpr_execution_reseter = reset;
  }

  /// pyramid type
  using PyramidType = PyramidAutomemSep<SepFltrColOP, SepFltrRowOP, DownSmplOP,
                                        RHS, SepFilterCol, SepFilterRow, Cols,
                                        Rows, LfType, LVL, Dp>;
  /// method get
  template <size_t N>
  PyramidLeafNode<PyramidType, N> get() {
    return PyramidLeafNode<PyramidType, N>(*this);
  }

  /// sub_expression_evaluation
  /// \brief This function is used to break the expression tree whenever
  /// necessary. The decision for breaking the tree will be determined based on
  /// the static parameter called SubExpressionEvaluationNeeded. When this is
  /// set to true, the sub_expression_evaluation is called recursively from the
  /// root of the tree. Each node based on their parent decision will decide to
  /// launch a kernel for itself. Also, they decide for each of their children
  /// whether or not to launch a kernel separately.
  /// template parameters:
  ///\tparam ForcedToExec : a boolean value representing the decision made by
  /// the parent of this node for launching a kernel.
  /// \tparam LC: is the column size of local memory required by Filter2D and
  /// DownSmplOP
  /// \tparam LR: is the row size of local memory required by Filter2D and
  /// DownSmplOP
  /// \tparam LCT: is the column size of workgroup
  /// \tparam LRT: is the row size of workgroup
  /// \tparam DeviceT: type representing the device
  /// function parameters:
  /// \param dev : the selected device for executing the expression
  /// \return LeafNode
  template <bool ForcedToExec, size_t LC, size_t LR, size_t LCT, size_t LRT,
            typename DeviceT>
  PyramidLeafNode<PyramidType, 0> inline sub_expression_evaluation(
      const DeviceT &dev) {
    // clearing the board
    auto eval_sub =
        rhs.template sub_expression_evaluation<false, LC, LR, LCT, LRT>(dev);

    PyramidExecuteAutoMemSep<
        Depth == 0, typename SepFltrColOP::OP, typename SepFltrRowOP::OP,
        typename DownSmplOP::OP, Cols, Rows, LeafType, Cols, 0, 1 + LVL, LC, LR,
        LCT, LRT, Depth, 0, LHSExpr, decltype(eval_sub), SepFilterCol,
        SepFilterRow, PyramidMem>::sub_execute(eval_sub, spFltrCol, spFltrRow,
                                               mem, dev);
    return get<0>();
  }
};
}  // internal

/// pyramid_auto_generate
/// \brief template deduction for PyramidAutomemGenSep
template <typename COP_C, typename COP_R, typename ROP, size_t Depth,
          typename RHS, typename SepFilterCol, typename SepFilterRow,
          typename... Params>
auto pyramid_down(RHS rhs, SepFilterCol spFltrCol, SepFilterRow spFltrRow)
    -> internal::PyramidAutomemSep<
        internal::LocalBinaryOp<COP_C, typename RHS::OutType,
                                typename SepFilterCol::OutType>,
        internal::LocalBinaryOp<COP_R, typename RHS::OutType,
                                typename SepFilterRow::OutType>,
        internal::LocalUnaryOp<ROP, typename RHS::OutType>, RHS, SepFilterCol,
        SepFilterRow, RHS::Type::Cols, RHS::Type::Rows, RHS::Type::LeafType,
        1 + RHS::Level, Depth> {
  return internal::PyramidAutomemSep<
      internal::LocalBinaryOp<COP_C, typename RHS::OutType,
                              typename SepFilterCol::OutType>,
      internal::LocalBinaryOp<COP_R, typename RHS::OutType,
                              typename SepFilterRow::OutType>,
      internal::LocalUnaryOp<ROP, typename RHS::OutType>, RHS, SepFilterCol,
      SepFilterRow, RHS::Type::Cols, RHS::Type::Rows, RHS::Type::LeafType,
      1 + RHS::Level, Depth>(rhs, spFltrCol, spFltrRow);
}
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EXPR_TREE_COMPLEX_OPS_PYRAMID_WITH_AUTO_MEM_SEP_HPP_

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

/// \file evaluator.hpp
/// \brief This file contains a collection of headers and forward declaration
/// for evaluating an expression tree.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVALUATOR_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVALUATOR_HPP_

namespace visioncpp {
namespace internal {
/// \struct GetGlobalRange
/// \brief GetGlobalRange is used to check the range when the halo is applied
///  template parameters
/// \tparam Halo is the halo used around the image
/// \tparam DimSize is the size of the dimension we want to check
template <size_t Halo, size_t DimSize>
struct GetGlobalRange {
  /// function get_global_range checks the range and pass the correct value as
  /// an index
  /// parameters:
  /// \param index is the passed index to be checked and corrected if needed
  /// \return size_t
  static size_t inline get_global_range(size_t index) {
    size_t val = index;
    if (val < Halo)
      val = 0;
    else if (val >= DimSize)
      val = DimSize - 1;
    else
      val -= Halo;
    return val;
  }
};
/// \brief specialisation of GetGlobalRange when the Halo is 0
template <size_t DimSize>
struct GetGlobalRange<0, DimSize> {
  /// function get_global_range checks the range and pass the correct value as
  /// an index
  /// parameters:
  /// \param index is the passed index to be checked and corrected if needed
  /// \return size_t
  static size_t inline get_global_range(size_t index) {
    return ((index < DimSize) ? index : DimSize - 1);
  }
};
/// \brief template deduction function for get_global_range
///  template parameters:
/// \tparam Halo is the halo used around the image
/// \tparam DimSize is the size of the dimension we want to check
/// function parameters:
/// \param index is the passed index to be checked and corrected if needed
/// \return size_t
template <size_t Halo, size_t DimSize>
static size_t inline get_global_range(size_t index) {
  return GetGlobalRange<Halo, DimSize>::get_global_range(index);
}
/// \struct Fill
/// \brief The Fill is used to load a rectangle neighbour area from
/// global memory to local memory. However, when the memory is constant or
/// located in device constant memory we do not create a load for them in shared
/// memory. LR and LC determines the valid size of local memory for the parent
/// of this function, the child can add its own valid size to it for its
/// calculation. This will happen when we have StnFilt
/// template parameters
/// \tparam Expr : the leaf node expression
/// \tparam Sc : the  sycl target representing the location of the buffer on the
/// device memory
/// \tparam Loc the coordinate needed to be accessed
/// \tparam Params... tuple of accessors
template <typename Expr, typename Loc, typename... Params>
struct Fill;
/// \struct Trait
/// \brief This struct is used to trait the value type inside the accessor
template <typename T>
struct Trait;
/// specialisation of the Trait class when the accessor is sycl accessor
template <typename elementType, int dimensions,
          cl::sycl::access::mode accessMode,
          cl::sycl::access::target accessTarget>
struct Trait<
    cl::sycl::accessor<elementType, dimensions, accessMode, accessTarget>> {
  using Type = elementType;
  static constexpr int Dim = dimensions;
  static constexpr size_t scope = ConvertToVisionScope<accessTarget>::scope;
};
/// specialisation of the Trait class when the accessor is on ConstMemory
template <typename T>
struct Trait<visioncpp::internal::ConstMemory<T>> {
  using Type = T;
  static constexpr size_t scope = scope::Global;
};

/// \struct Index_Finder This struct is used to find the index required to
/// access the accessor inside the buffer.
/// template parameters
/// \tparam N the location of the global memory
/// \tparam Index the location of the local memory if exists
/// \tparam LeafType the type of the memory
/// \tparam Sc the location of the buffer on the device memory
template <size_t N, size_t Indx, size_t LeafType, size_t Sc>
struct Index_Finder {
  static constexpr size_t Index = Indx;
};
/// specialisation of the Index_Finder when the memory_type is a constant variable
template <size_t N, size_t Indx, size_t Sc>
struct Index_Finder<N, Indx, memory_type::Const, Sc> {
  static constexpr size_t Index = N;
};

/// specialisation of the Index_Finder when the memory located on device
/// constant memory
template <size_t N, size_t Indx, size_t LeafType>
struct Index_Finder<N, Indx, LeafType, scope::Constant> {
  static constexpr size_t Index = N;
};

/// \brief the root of the expression tree. It is used to avoid extra creation
/// of the shared memory for the LHS of the root node.
/// The are two types of nodes that can be the root of the expression tree.
/// ParallelCopy and assign.
/// An expression tree can be evaluated in three ways, depending on the type of
/// operation it represents. A global operation node represents that the
/// operation requires to access the entire input or non-linear area of input to
/// calculate an element of the output; a neighbour operation node requires to
/// access a linear area of the input in order to calculate the output; and a
/// point operation node requires a corresponding elements of the input to
/// calculate an elements of the output. Therefore, we have tree types of
/// Operations. GlobalNeighbourOP, NeighbourOP, PointOP. Thus, 6 specialisation
/// of Evaluator template struct has been provided. In an expression tree with
/// the mixed types of nodes following rules will be applied. \n
/// 1) If an expression tree contains a combination of point and neighbour
/// operations nodes, the neighbour operation will be used for the evaluation of
/// the expression tree without splitting the expression tree into small
/// sub-trees. In this case only one kernel will be executed. \n
/// 2)If an expression tree has global operation node in the middle of the tree,
/// the expression tree will be split in to 3 sub-trees.  One before the
/// global operation node; one is the global operation node; and one is the
/// nodes after global expression node. \n
/// 3) If the global operation is the root of the expression tree, the
/// expression will be split in to two sub-trees. \n
/// 4) If the child of the global operation node is a leaf node, the child will
/// not be split in to a sub-tree.  The eval struct has been
/// specialised
/// based on the operation types of the expression tree, in order to call the
/// appropriate function.\n
/// These specialisation of Evaluator class are located in
/// eval_root/eval_assign.hpp and eval_root/eval_partial_assign.hpp
/// files.
/// Typical usage:
/// \code
/// \endcode
///
/// Template parameters:
/// \param Offset: the starting offset of the output shared
/// \param memories inside the input tuple.
/// \param Output_Index: the required step from offset in order to
/// \param access the output memory of the current expression
/// \param LHS: the left-hand size expression
/// \param RHS: then right-hand size expression
/// \param internal::ops_category: the type of the operation
/// \param LC: the shared memory column size
/// \param LR: the shared memory row size
/// \param Cols: output memory column size
/// \param Rows: output memory row size
/// \param LVL: the depth of the subexpression root in the expression tree
/// \param LfType: the type of the output memory
/// \param Loc: Coordinate of accessing a particular location
/// \param Params: the input/output memories
///
/// eval function parameters:
/// \param  cOffset Coordinate of accessing a particular location.
/// \param  t: tuple of input/output memories
///
/// \returns void.
template <size_t OPT, size_t Output_Index, size_t Offset, size_t LC, size_t LR,
          typename Expr, typename Loc, typename... Params>
struct Evaluator;

/// \struct EvalExpr
/// \brief is used to convert the static expression tree to the runtime the
/// expression tree when it is not a root of an expression tree. It is used to
/// calculate the runtime evaluation of the specified operators for each node.
/// The runtime Executor expression tree can be specialised per device through
/// template parameters. Each primary node in an expression tree will have an
/// equivalent node in the evaluation expression tree. An expression tree can be
/// evaluated in three ways, depending on the type of
/// operation it represents.
/// A point operation node is evaluated through eval_point function; a
/// neighbour
/// operation node is evaluated by eval_neighbour; and a global operation node
/// is evaluated by eval_global_neighbour. The specialisation of EvalExpr
/// located on eval_expression folder.
/// Template parameters
///         \param Expr: the expression node required to be executed.
///         \param Loc: Coordinate of accessing a particular location
///         \param params: the input/output memories
/// eval_point function: \brief point operation evaluation.
/// \param  cOffset: coordinate of accessing a particular location.
/// \param  t: tuple of the input/output memories
/// \return the calculated memory element (pixel)
/// \n
/// eval_neighbour function: \brief neighbour operation evaluation.
/// \param IsRoot: check whether or not this node should use the global memory
/// as an output memory or a local memory. This is used to avoid the last local
/// memory between assign and the immediate RHS expression in Assign.
/// \param Halo_Top: pass the top side value of halo for row of the local memory
/// \param Halo_Left: pass the left side value of halo for column of the local
/// memory
/// \param Halo_Butt: pass the bottom side value of halo for row of the local
/// memory
/// \param Halo_Right: pass the right side value of halo for column of the local
/// memory
/// \param Offset: determines the starting location of the local output memory
/// in the input tuple.
/// \param Index: represent the distance of the local memory in the tuple for
/// this node from the Offset.
/// \param LC: determines the local memory column size.
/// \param LR: determines the local memory row size.
/// \return the reference to the output memory block calculated the value.

/// eval_global_neighbour function: \brief global operation evaluation
/// \param IsRoot: check whether or not this node should use the global memory
/// as an output memory or a local memory. This is used to avoid the last local
/// memory between assign and the immediate RHS expression in Assign.
/// \param Offset: determines the starting location of the local output memory
/// in the input tuple.
/// \param Index: represent the distance of the local memory in the tuple for
/// this node from the Offset.
/// \param LC: determines the local memory column size.
/// \param LR: determines the local memory row size.
/// \return the reference to the output memory block calculated the value.

template <typename Expr, typename Loc, typename... Params>
struct EvalExpr;

/// \struct OutputLocation
/// \brief This is used to find whether a node should use a global memory
/// output or a local memory output is created for that node. When the node is
/// the immediate child of the root (Assign or Partial Assign) we do not create
/// the local shared memory for it and directly save the result in the global
/// memory. In this case we can avoid creation of unnecessary shared memory.
/// template parameters
/// \tparam IsRoot boolean value represent if the expression is the immediate
/// child of a root node
/// \tparam OutOffset: the offset for location of the local memory if it exists
template <bool IsRoot, size_t OutOffset>
struct OutputLocation {
  static constexpr size_t ID = OutOffset;
};
/// \brief specialisation of the OutputLocation when the node is the immediate
/// child of the root node
template <size_t OutOffset>
struct OutputLocation<true, OutOffset> {
  static constexpr size_t ID = 0;
};
/// \brief template deduction for Fill struct.
template <size_t Halo_Top, size_t Halo_Left, size_t Halo_Butt,
          size_t Halo_Right, size_t Offset, size_t LC, size_t LR, size_t Sc,
          typename Expr, typename Loc, typename... Params>
static void fill_local_neighbour(
    Loc &cOffset, const internal::tools::tuple::Tuple<Params...> &t);

/// \brief deduction function for Evaluator struct.
template <size_t Offset, size_t LC, size_t LR, typename Expr, typename Loc,
          typename... Params>
inline void eval(Loc &cOffset, const tools::tuple::Tuple<Params...> &t) {
  /// \brief This will count N+1 where the number of memory is N. However the N+1 is
  /// useless as it does not exist and is going to be replaced by the final output
  /// node in the expression tree trough OutputLocation struct.
  constexpr size_t Index = LocalMemCount<Expr::ND_Category, Expr>::Count;
  Evaluator<Expr::Operation_type, Index, Offset, LC, LR, Expr, Loc,
            Params...>::eval(cOffset, t);
}
}  // internal
}  // visioncpp

#include "eval_assign/eval_assign.hpp"
#include "eval_assign/eval_assign_partial.hpp"
#include "eval_expression/eval_expression.hpp"
#include "load_pattern/square_pattern.hpp"
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_EVALUATOR_EVALUATOR_HPP_

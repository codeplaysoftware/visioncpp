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

/// \file sycl_device.hpp
/// \brief This fill contains features related to sycl devices.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_DEVICE_SYCL_DEVICE_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_DEVICE_SYCL_DEVICE_HPP_

namespace visioncpp {
namespace internal {

/// \brief specialisation Device_ for sycl
/// \tparam device type supported by sycl
template <device dv>
class Device_<backend::sycl, dv> {
  using QueueType = cl::sycl::queue;
  using DevType = typename DeviceSelector<dv>::Type;

 private:
  mutable QueueType dev;

 public:
  Device_()
      : dev(QueueType(DevType(), [=](cl::sycl::exception_list l) {
          for (const auto &e : l) {
            try {
              std::rethrow_exception(e);
            } catch (cl::sycl::cl_exception e) {
              std::cout << e.get_cl_error_message() << std::endl;
              std::cout << e.get_cl_code() << std::endl;
            } catch (cl::sycl::exception e) {
              std::cout << e.what() << std::endl;
            }
          }
        })) {}
  template <size_t LC, size_t LR, size_t CGT, size_t RGT, size_t CLT,
            size_t RLT, typename Expr>
  void execute(Expr &expr) const {
    /// generating the short class name for the AMD gpu
    constexpr size_t TotalLeaves = LeafCount<Expr::ND_Category, Expr>::Count;
    /// replacing the the leaf node in the expression tree with a placeholder
    /// number
    using placeHolderExprType =
        typename MakePlaceHolderExprHelper<Expr::ND_Category, Expr,
                                           TotalLeaves - 1>::Type;

    /// submitting the lambda expression to the sycl queue.
    dev.submit([&](cl::sycl::handler &cgh) {

      /// creating global accessors on all input output buffers
      auto global_accessor_tuple = extract_accessors(cgh, expr);
      /// create the tuple of local output accessor
      auto device_only_accessor_tuple =
          create_local_accessors<LC, LR, Expr>(cgh);
      /// starting point of local tuples
      constexpr size_t Output_offset =
          tools::tuple::size(global_accessor_tuple);
      /// merge it with all the other existing tuples
      auto device_tuple = tools::tuple::append(global_accessor_tuple,
                                               device_only_accessor_tuple);
      /// submitting the kernel lambda to the parallel

      cgh.parallel_for<Expr>(
          cl::sycl::nd_range<Expr::Type::Dim>(
              visioncpp::internal::get_range<Expr::Type::Dim>(RGT, CGT),
              visioncpp::internal::get_range<Expr::Type::Dim>(RLT, CLT)),
          [=](cl::sycl::nd_item<Expr::Type::Dim> itemID) {
            /// creating the index access for each thread
            auto cOffset = visioncpp::internal::memLocation<LC, LR>(itemID);

            /// creating the eval expression for evaluating the expression
            /// tree. The output now moved to the front so the Output_offset
            /// should be reduced by one.
            eval<Output_offset, LC, LR, placeHolderExprType>(cOffset,
                                                             device_tuple);
          });
    });
    dev.throw_asynchronous();
  }
};
}
}
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_DEVICE_SYCL_DEVICE_HPP_
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

/// \file mem_prop.hpp
/// \brief Series of pixel convert functions.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEM_PROP_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEM_PROP_HPP_

namespace visioncpp {
namespace internal {

/// \brief This file is used to detect the ChannelType ElementCategory {basic or
/// struct}, and the channel size of a row input data
template <typename ElementTp>
struct MemoryProperties {
  static constexpr size_t ElementCategory = element_category::Struct;
  using ChannelType = typename ElementTp::data_type;
  static constexpr size_t ChannelSize = ElementTp::elements;
};

/// \brief Specialisation of the MemoryProperties when the output is unsigned
/// char
template <>
struct MemoryProperties<unsigned char> {
  static constexpr size_t ElementCategory = element_category::Basic;
  using ChannelType = unsigned char;
  static constexpr size_t ChannelSize = 1;
};

/// \brief Specialisation of the MemoryProperties when the output is char
template <>
struct MemoryProperties<char> {
  static constexpr size_t ElementCategory = element_category::Basic;
  using ChannelType = char;
  static constexpr size_t ChannelSize = 1;
};

/// \brief Specialisation of the MemoryProperties when the output is unsigned
/// short
template <>
struct MemoryProperties<unsigned short> {
  static constexpr size_t ElementCategory = element_category::Basic;
  using ChannelType = unsigned short;
  static constexpr size_t ChannelSize = 1;
};

/// \brief Specialisation of the MemoryProperties when the output is short
template <>
struct MemoryProperties<short> {
  static constexpr size_t ElementCategory = element_category::Basic;
  using ChannelType = short;
  static constexpr size_t ChannelSize = 1;
};

/// \brief Specialisation of the MemoryProperties when the output is unsigned
/// int
template <>
struct MemoryProperties<unsigned int> {
  static constexpr size_t ElementCategory = element_category::Basic;
  using ChannelType = unsigned int;
  static constexpr size_t ChannelSize = 1;
};

/// \brief Specialisation of the MemoryProperties when the output is int
template <>
struct MemoryProperties<int> {
  static constexpr size_t ElementCategory = element_category::Basic;
  using ChannelType = int;
  static constexpr size_t ChannelSize = 1;
};

/// \brief Specialisation of the MemoryProperties when the output is float
template <>
struct MemoryProperties<float> {
  static constexpr size_t ElementCategory = element_category::Basic;
  using ChannelType = float;
  static constexpr size_t ChannelSize = 1;
};
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_MEMORY_MEM_PROP_HPP_

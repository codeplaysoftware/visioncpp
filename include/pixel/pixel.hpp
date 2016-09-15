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

/// \file pixel.hpp
/// \brief This file contains definition of underlying pixel types used in
/// VisionCpp.
///  Generic form of the pixel is as follow: \n
/// {F/U/S}{SIZE_OF_CHANNEL}C{NUMBER_OF_CHANNELS} \n
/// F32C4 - represents float[4] \n
/// U8C3  - represents unsigned char[3] \n

#ifndef VISIONCPP_INCLUDE_PIXEL_PIXEL_HPP_
#define VISIONCPP_INCLUDE_PIXEL_PIXEL_HPP_

#include "../framework/tools/tuple.hpp"

namespace visioncpp {
namespace internal {
template <bool conds, size_t K, typename PixelType, typename... Params>
struct AssignValueToArray {
  /// \brief Assigns Value To Array
  static void avta(PixelType &dt,
                   visioncpp::internal::tools::tuple::Tuple<Params...> &t) {
    dt[K] = visioncpp::internal::tools::tuple::get<K>(t);
    AssignValueToArray<K + 1 != sizeof...(Params), K + 1, PixelType,
                       Params...>::avta(dt, t);
  }
};

template <size_t K, typename PixelType, typename... Params>
struct AssignValueToArray<false, K, PixelType, Params...> {
  /// \brief Assigns Value To Array
  static void avta(PixelType &dt,
                   visioncpp::internal::tools::tuple::Tuple<Params...> &t) {}
};
}
/// \brief Contains VisionCpp pixel type definitions.
namespace pixel {
#define REGISTER_OPERATORS(Op, T)                     \
  template <typename RHSScalar>                       \
  T &operator Op##=(const RHSScalar &val) {           \
    for (int i = 0; i < elements; i++) {              \
      m_data[i] Op## = val;                           \
    }                                                 \
    return *this;                                     \
  }                                                   \
  template <typename RHSScalar>                       \
  friend T operator Op(T lhs, const RHSScalar &rhs) { \
    for (int i = 0; i < elements; i++) {              \
      lhs[i] Op## = rhs;                              \
    }                                                 \
    return lhs;                                       \
  }                                                   \
  T &operator Op##=(const T &val) {                   \
    for (int i = 0; i < elements; i++) {              \
      m_data[i] Op## = val[i];                        \
    }                                                 \
    return *this;                                     \
  }                                                   \
  friend T operator Op(T lhs, const T &rhs) {         \
    for (int i = 0; i < elements; i++) {              \
      lhs[i] Op## = rhs[i];                           \
    }                                                 \
    return lhs;                                       \
  }

template <typename LHSScalar, size_t Channels>
struct Storage {
  typedef LHSScalar data_type;
  constexpr static size_t elements = Channels;
  LHSScalar m_data[elements];
  data_type operator[](unsigned int idx) const { return m_data[idx]; }
  data_type &operator[](unsigned int idx) { return m_data[idx]; }
  REGISTER_OPERATORS(+, Storage)
  REGISTER_OPERATORS(-, Storage)
  REGISTER_OPERATORS(/, Storage)
  REGISTER_OPERATORS(*, Storage)
  template <typename... P>
  Storage(P... p) {
    auto tp = visioncpp::internal::tools::tuple::make_tuple(p...);
    internal::AssignValueToArray<0 != sizeof...(P), 0, decltype(m_data), P...>::avta(
        m_data, tp);
  }
};

/// \struct F32C1
/// \brief This struct is generalisation for three channels float that is
/// perfect for storing pixels of R.
typedef Storage<float, 1> F32C1;

/// \struct F32C2
/// \brief This struct is generalisation for three channels float that is
/// perfect for storing pixels of RG and permutations.
typedef Storage<float, 2> F32C2;

/// \struct F32C3
/// \brief This struct is generalisation for three channels float that is
/// perfect for storing pixels of RGB and permutations.
typedef Storage<float, 3> F32C3;

/// \struct F32C4
/// \brief This struct is generalisation for three channels float that is
/// perfect for storing pixels of RGBA and permutations.
typedef Storage<float, 4> F32C4;

/// \struct U8C1
/// \brief This struct is generalisation for three channels float that is
/// perfect for storing pixels of R.
typedef Storage<unsigned char, 1> U8C1;

/// \struct U8C2
/// \brief This struct is generalisation for three channels unsigned char that
/// is perfect for storing pixels of RG and permutations.
typedef Storage<unsigned char, 2> U8C2;

/// \struct U8C3
/// \brief This struct is generalisation for three channels unsigned char that
/// is perfect for storing pixels of RGB and permutations.
typedef Storage<unsigned char, 3> U8C3;

/// \struct U8C4
/// \brief This struct is generalisation for three channels unsigned char that
/// is perfect for storing pixels of RGBA and permutations.
typedef Storage<unsigned char, 4> U8C4;

}  // end of pixel
}  // end of visionCPP
#endif  // VISIONCPP_INCLUDE_PIXEL_PIXEL_HPP_

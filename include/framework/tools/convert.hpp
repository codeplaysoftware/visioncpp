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

/// \file convert.hpp
/// \brief Series of pixel convert functions.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_TOOLS_CONVERT_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_TOOLS_CONVERT_HPP_

namespace visioncpp {
namespace internal {
/// \brief Internal tools scope.
namespace tools {
/// \brief This struct is used to convert the provide struct to flot4, uint4,
/// int4 memory. Also it s used to propagate all the channels of an element with
/// one primary value. The former is used when we have an image while the latter is
/// used for broadcast function. When both types are the same it does nothing
/// but returns the input.
/// template parameters:
/// \tparam T is the input type to be converted
template <typename T>
struct Convertor {
  /// function convert
  /// \brief Convert the input type to the output type. When both types are the
  ///  same it does nothing but returns the input.
  /// parameters:
  /// \param t  Input type to be converted
  /// \return T
  static inline T convert(T t) { return t; }
};

/// \brief Specialisation of Convertor when the output is float4
template <>
struct Convertor<cl::sycl::float4> {
  /// function convert
  /// \brief Convert the F32C3 type to the cl::sycl::float4 type.
  /// parameters:
  /// \param t  Input type  to be converted
  /// \return float4
  static inline cl::sycl::float4 convert(visioncpp::pixel::F32C3 t) {
    return cl::sycl::float4(t[0], t[1], t[2], 0.0f);
  }

  /// function convert
  /// \brief Convert the F32C4 input type to the cl::sycl::float4 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return float4
  static inline cl::sycl::float4 convert(visioncpp::pixel::F32C4 t) {
    return cl::sycl::float4(t[0], t[1], t[2], t[3]);
  }

  /// function convert
  /// \brief Convert the U8C3 input type to the cl::sycl::float4 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return float4
  static inline cl::sycl::float4 convert(visioncpp::pixel::U8C3 t) {
    return cl::sycl::float4(static_cast<float>(t[0]), static_cast<float>(t[1]),
                            static_cast<float>(t[2]), 0.0f);
  }

  /// function convert
  /// \brief Convert the U8C4 input type to the cl::sycl::float4 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return float4
  static inline cl::sycl::float4 convert(visioncpp::pixel::U8C4 t) {
    return cl::sycl::float4(static_cast<float>(t[0]), static_cast<float>(t[1]),
                            static_cast<float>(t[2]), static_cast<float>(t[3]));
  }

  /// function convert
  /// \brief Convert the float input type to the cl::sycl::float4 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return float4
  static inline cl::sycl::float4 convert(float t) {
    return cl::sycl::float4(t, t, t, t);
  }
};
/// \brief specialisation of Convertor when the output is int4
template <>
struct Convertor<cl::sycl::int4> {
  /// function convert
  /// \brief Convert the F32C3 input type to the cl::sycl::int4 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return int4
  static inline cl::sycl::int4 convert(visioncpp::pixel::F32C3 t) {
    return cl::sycl::int4(static_cast<int>(t[0]), static_cast<int>(t[1]),
                          static_cast<int>(t[2]), 0);
  }

  /// function convert
  /// \brief Convert the F32C4 input type to the cl::sycl::int4 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return int4
  static inline cl::sycl::int4 convert(visioncpp::pixel::F32C4 t) {
    return cl::sycl::int4(static_cast<int>(t[0]), static_cast<int>(t[1]),
                          static_cast<int>(t[2]), static_cast<int>(t[3]));
  }

  /// function convert
  /// \brief Convert the U8C3 input type to the cl::sycl::int4 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return int4
  static inline cl::sycl::int4 convert(visioncpp::pixel::U8C3 t) {
    return cl::sycl::int4(static_cast<int>(t[0]), static_cast<int>(t[1]),
                          static_cast<int>(t[2]), 0);
  }

  /// function convert
  /// \brief Convert the U8C4 input type to the cl::sycl::int4 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return int4
  static inline cl::sycl::int4 convert(visioncpp::pixel::U8C4 t) {
    return cl::sycl::int4(static_cast<int>(t[0]), static_cast<int>(t[1]),
                          static_cast<int>(t[2]), static_cast<int>(t[3]));
  }

  /// function convert
  /// \brief Convert the int input type to the cl::sycl::int4 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return int4
  static inline cl::sycl::int4 convert(int t) {
    return cl::sycl::int4(t, t, t, t);
  }
};
/// \brief specialisation of the Convertor when the output is uint4
template <>
struct Convertor<cl::sycl::uint4> {
  /// function convert
  /// \brief Convert the F32C3 input type to the cl::sycl::uint4 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return uint4
  static inline cl::sycl::uint4 convert(visioncpp::pixel::F32C3 t) {
    return cl::sycl::uint4(static_cast<unsigned int>(t[0]),
                           static_cast<unsigned int>(t[1]),
                           static_cast<unsigned int>(t[2]), 0);
  }

  /// function convert
  /// \brief Convert the F32C4 input type to the cl::sycl::uint4 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return uint4
  static inline cl::sycl::uint4 convert(visioncpp::pixel::F32C4 t) {
    return cl::sycl::uint4(
        static_cast<unsigned int>(t[0]), static_cast<unsigned int>(t[1]),
        static_cast<unsigned int>(t[2]), static_cast<unsigned int>(t[3]));
  }

  /// function convert
  /// \brief Convert the U8C3 input type to the cl::sycl::uint4 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return uint4
  static inline cl::sycl::uint4 convert(visioncpp::pixel::U8C3 t) {
    return cl::sycl::uint4(static_cast<unsigned int>(t[0]),
                           static_cast<unsigned int>(t[1]),
                           static_cast<unsigned int>(t[2]), 0);
  }

  /// function convert
  /// \brief Convert the U8C4 input type to the cl::sycl::uint4 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return uint4
  static inline cl::sycl::uint4 convert(visioncpp::pixel::U8C4 t) {
    return cl::sycl::uint4(
        static_cast<unsigned int>(t[0]), static_cast<unsigned int>(t[1]),
        static_cast<unsigned int>(t[2]), static_cast<unsigned int>(t[3]));
  }

  /// function convert
  /// \brief Convert the unsigned int input type to the cl::sycl::uint4 output
  /// type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return uint4
  static inline cl::sycl::uint4 convert(unsigned int t) {
    return cl::sycl::uint4(t, t, t, t);
  }

  /// function convert
  /// \brief Convert the unsigned char input type to the cl::sycl::uint4 output
  /// type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return uint4
  static inline cl::sycl::uint4 convert(unsigned char t) {
    return cl::sycl::uint4(t, t, t, t);
  }
};
/// \brief specialisation of the Convertor when the output is F32C3
template <>
struct Convertor<visioncpp::pixel::F32C3> {
  /// function convert
  /// \brief Convert the cl::sycl::float4 input type to the F32C3 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return F32C3
  static inline visioncpp::pixel::F32C3 convert(cl::sycl::float4 t) {
    return visioncpp::pixel::F32C3(t.x(), t.y(), t.z());
  }

  /// function convert
  /// \brief Returns the input type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return F32C3
  static inline visioncpp::pixel::F32C3 convert(visioncpp::pixel::F32C3 t) {
    return t;
  }

  /// function convert
  /// \brief Convert the cl::sycl::int4 input type to the F32C3 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return F32C3
  static inline visioncpp::pixel::F32C3 convert(cl::sycl::int4 t) {
    return visioncpp::pixel::F32C3(static_cast<float>(t.x()),
                                   static_cast<float>(t.y()),
                                   static_cast<float>(t.z()));
  }

  /// function convert
  /// \brief Convert the cl::sycl::uint4 input type to the F32C3 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return F32C3
  static inline visioncpp::pixel::F32C3 convert(cl::sycl::uint4 t) {
    return visioncpp::pixel::F32C3(static_cast<float>(t.x()),
                                   static_cast<float>(t.y()),
                                   static_cast<float>(t.z()));
  }

  /// function convert
  /// \brief Convert the float input type to the F32C3 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return F32C3
  static inline visioncpp::pixel::F32C3 convert(float t) {
    return visioncpp::pixel::F32C3(t, t, t);
  }
};
/// \brief specialisation of the Convertor when the output is F32C4
template <>
struct Convertor<visioncpp::pixel::F32C4> {
  /// function convert
  /// \brief Convert the cl::sycl::float4 input type to the F32C4 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return F32C4
  static inline visioncpp::pixel::F32C4 convert(cl::sycl::float4 t) {
    return visioncpp::pixel::F32C4(t.x(), t.y(), t.z(), t.w());
  }

  /// function convert
  /// \brief Returns the F32C4 input type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return F32C4
  static inline visioncpp::pixel::F32C4 convert(visioncpp::pixel::F32C4 t) {
    return t;
  }

  /// function convert
  /// \brief Convert the cl::sycl::int4 input type to the F32C4 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return F32C4
  static inline visioncpp::pixel::F32C4 convert(cl::sycl::int4 t) {
    return visioncpp::pixel::F32C4(
        static_cast<float>(t.x()), static_cast<float>(t.y()),
        static_cast<float>(t.z()), static_cast<float>(t.w()));
  }

  /// function convert
  /// \brief Convert the cl::sycl::uint4 input type to the F32C4 output type.
  /// parameters:
  /// \param t  input type to be converted
  /// \return F32C4
  static inline visioncpp::pixel::F32C4 convert(cl::sycl::uint4 t) {
    return visioncpp::pixel::F32C4(
        static_cast<float>(t.x()), static_cast<float>(t.y()),
        static_cast<float>(t.z()), static_cast<float>(t.w()));
  }

  /// function convert
  /// \brief Convert the float input type to the F32C4 output type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return F32C4
  static inline visioncpp::pixel::F32C4 convert(float t) {
    return visioncpp::pixel::F32C4(t, t, t, t);
  }
};
/// \brief specialisation of the Convertor when the output is U8C3
template <>
struct Convertor<visioncpp::pixel::U8C3> {
  /// function convert
  /// \brief Convert the cl::sycl::float4 input type to the U8C3 output type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return U8C3
  static inline visioncpp::pixel::U8C3 convert(cl::sycl::float4 t) {
    return visioncpp::pixel::U8C3(t.x(), t.y(), t.z());
  }

  /// function convert
  /// \brief Convert the unsigned char input type to the U8C3 output type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return U8C3
  static inline visioncpp::pixel::U8C3 convert(unsigned char t) {
    return visioncpp::pixel::U8C3(t, t, t);
  }

  /// function convert
  /// \brief Returns the U8C3 input type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return U8C3
  static inline visioncpp::pixel::U8C3 convert(visioncpp::pixel::U8C3 t) {
    return t;
  }

  /// function convert
  /// \brief Convert the cl::sycl::int4 input type to the U8C3 output type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return U8C3
  static inline visioncpp::pixel::U8C3 convert(cl::sycl::int4 t) {
    return visioncpp::pixel::U8C3(static_cast<unsigned char>(t.x()),
                                  static_cast<unsigned char>(t.y()),
                                  static_cast<unsigned char>(t.z()));
  }

  /// function convert
  /// \brief Convert the cl::sycl::uint4 input type to the U8C3 output type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return U8C3
  static inline visioncpp::pixel::U8C3 convert(cl::sycl::uint4 t) {
    return visioncpp::pixel::U8C3(static_cast<unsigned char>(t.x()),
                                  static_cast<unsigned char>(t.y()),
                                  static_cast<unsigned char>(t.z()));
  }
};
/// \brief specialisation of the Convertor when the output is U8C4
template <>
struct Convertor<visioncpp::pixel::U8C4> {
  /// function convert
  /// \brief Convert the cl::sycl::float4 input type to the U8C4 output type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return U8C4
  static inline visioncpp::pixel::U8C4 convert(cl::sycl::float4 t) {
    return visioncpp::pixel::U8C4(t.x(), t.y(), t.z(), t.w());
  }

  /// function convert
  /// \brief Convert the unsigned char input type to the U8C4 output type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return U8C4
  static inline visioncpp::pixel::U8C4 convert(unsigned char t) {
    return visioncpp::pixel::U8C4(t, t, t);
  }

  /// function convert
  /// \brief Returns the U8C4 input type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return U8C4
  static inline visioncpp::pixel::U8C4 convert(visioncpp::pixel::U8C4 t) {
    return t;
  }

  /// function convert
  /// \brief Convert the cl::sycl::int4 input type to the U8C4 output type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return U8C4
  static inline visioncpp::pixel::U8C4 convert(cl::sycl::int4 t) {
    return visioncpp::pixel::U8C4(
        static_cast<unsigned char>(t.x()), static_cast<unsigned char>(t.y()),
        static_cast<unsigned char>(t.z()), static_cast<unsigned char>(t.w()));
  }

  /// function convert
  /// \brief Convert the cl::sycl::uint4 input type to the U8C4 output type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return U8C4
  static inline visioncpp::pixel::U8C4 convert(cl::sycl::uint4 t) {
    return visioncpp::pixel::U8C4(
        static_cast<unsigned char>(t.x()), static_cast<unsigned char>(t.y()),
        static_cast<unsigned char>(t.z()), static_cast<unsigned char>(t.w()));
  }
};
/// \brief specialisation of the Convertor when the output is unsigned char
template <>
struct Convertor<unsigned char> {
  /// function convert
  /// \brief Convert the cl::sycl::uint4 input type to the unsigned char output
  /// type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return unsigned char
  static inline unsigned char convert(cl::sycl::uint4 t) { return t.x(); }

  /// function convert
  /// \brief Returns the unsigned char input type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return unsigned char
  static inline unsigned char convert(unsigned char t) { return t; }
};
/// \brief specialisation of the Convertor when the output is char
template <>
struct Convertor<char> {
  /// function convert
  /// \brief Convert the cl::sycl::int4 input type to the char output type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return char
  static inline char convert(cl::sycl::int4 t) { return t.x(); }

  /// function convert
  /// \brief Convert the input type to the output type. When both types are the
  ///  same it does nothing but returns the input.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return char
  static inline short convert(char t) { return t; }
};
/// \brief specialisation of the Convertor when the output is unsigned short
template <>
struct Convertor<unsigned short> {
  /// function convert
  /// \brief Convert the cl::sycl::uint4 t input type to the unsigned short
  /// output type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return unsigned short
  static inline unsigned short convert(cl::sycl::uint4 t) { return t.x(); }

  /// function convert
  /// \brief Convert the input type to the output type. When both types are the
  ///  same it does nothing but returns the input.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return unsigned short
  static inline unsigned short convert(unsigned short t) { return t; }
};
/// \brief specialisation of the Convertor when the output is short
template <>
struct Convertor<short> {
  /// function convert
  /// \brief Convert the cl::sycl::int4 input type to the short output type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return short
  static inline short convert(cl::sycl::int4 t) { return t.x(); }

  /// function convert
  /// \brief Returns the short input.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return short
  static inline short convert(short t) { return t; }
};
/// \brief specialisation of the Convertor when the output is unsigned int
template <>
struct Convertor<unsigned int> {
  /// function convert
  /// \brief Convert the cl::sycl::uint4 input type to the int output type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return unsigned int
  static inline int convert(cl::sycl::uint4 t) { return t.x(); }

  /// function convert
  /// \brief Returns the unsigned int input type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return unsigned int
  static inline unsigned int convert(unsigned int t) { return t; }
};
/// \brief specialisation of the Convertor when the output is int
template <>
struct Convertor<int> {
  /// function convert
  /// \brief Convert the cl::sycl::int4 input type to the int output type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return int
  static inline int convert(cl::sycl::int4 t) { return t.x(); }

  /// function convert
  /// \brief Returns the int input type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return int
  static inline int convert(int t) { return t; }
};
/// \brief specialisation of the Convertor when the output is float
template <>
struct Convertor<float> {
  /// function convert
  /// \brief Convert the cl::sycl::float4 input type to the float output type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return float
  static inline float convert(cl::sycl::float4 t) { return t.x(); }

  /// function convert
  /// \brief Returns the float input type.
  /// parameters:
  /// \param t  input type needed to be converted
  /// \return float
  static inline float convert(float t) { return t; }
};

/// function convert
/// \brief template deduction for Convertor struct
/// template parameters
/// \tparam T1 the output type
/// \tparam T2 the input type
/// function parameters
/// \param x: the input pixel
/// \return T1
template <typename T1, typename T2>
inline T1 convert(T2 x) {
  return Convertor<T1>::convert(x);
};
}  // tools
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_TOOLS_CONVERT_HPP_

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

/// \file type_dereferencer.hpp
/// \brief These methods are used to remove all the & const and * from  a type.

#ifndef VISIONCPP_INCLUDE_FRAMEWORK_TOOLS_TYPE_DEREFERENCER_HPP_
#define VISIONCPP_INCLUDE_FRAMEWORK_TOOLS_TYPE_DEREFERENCER_HPP_

namespace visioncpp {
namespace internal {
namespace tools {
/// \struct RemoveAll
/// \brief These methods are used to remove all the & const and * from  a type.
/// template parameters
/// \tparam T : the type we are interested in
template <typename T>
struct RemoveAll {
  typedef T Type;
};
/// specialisation of RemoveAll when the type contains &
template <typename T>
struct RemoveAll<T &> {
  typedef typename RemoveAll<T>::Type Type;
};
/// specialisation of RemoveAll when the type contains *
template <typename T>
struct RemoveAll<T *> {
  typedef typename RemoveAll<T>::Type Type;
};
/// specialisation of RemoveAll when the type contains const
template <typename T>
struct RemoveAll<const T> {
  typedef typename RemoveAll<T>::Type Type;
};

/// specialisation of RemoveAll when the type contains const and &
template <typename T>
struct RemoveAll<const T &> {
  typedef typename RemoveAll<T>::Type Type;
};
/// specialisation of RemoveAll when the type contains volatile
template <typename T>
struct RemoveAll<T volatile> {
  typedef typename RemoveAll<T>::Type Type;
};
/// specialisation of RemoveAll when the type contains const volatile
template <typename T>
struct RemoveAll<T const volatile> {
  typedef typename RemoveAll<T>::Type Type;
};
/// specialisation of RemoveAll when the type contains const and *
template <typename T>
struct RemoveAll<const T *> {
  typedef typename RemoveAll<T>::Type Type;
};
}  // tools
}  // internal
}  // visioncpp
#endif  // VISIONCPP_INCLUDE_FRAMEWORK_TOOLS_TYPE_DEREFERENCER_HPP_

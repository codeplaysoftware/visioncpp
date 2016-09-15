// This file is part of VisionCPP, a lightweight C++ template library
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

/// \file OP_Median.hpp
/// \brief It applies the median filter in an image. It gets the median value
/// of a neighbour after sorted

namespace visioncpp {
/// \brief User defined functionality for the kernel
namespace custom {
template <typename T>
void bubbleSort(T &a, int N) {
  bool swapp = true;
  while (swapp) {
    swapp = false;
    for (size_t i = 0; i < N - 1; i++) {
      if (a[i] > a[i + 1]) {
        a[i] += a[i + 1];
        a[i + 1] = a[i] - a[i + 1];
        a[i] -= a[i + 1];
        swapp = true;
      }
    }
  }
}
}

/// \brief This functor implements a median filter
struct OP_Median {
  /// \brief This functor implements median filter
  /// \param nbr - Input image
  /// \return float - Returns the image with median filter applied
  template <typename T>
  T operator()(T nbr) {
    int size = 5;
    int bound = size / 2;
    float v[25];
    int k = 0;
    for (int i = -bound; i <= bound; i++) {
      for (int j = -bound; j <= bound; j++) {
        v[k++] = nbr.at(nbr.I_c + i, nbr.I_r + j);
      }
    }
    custom::bubbleSort(v, size * size);
    return v[size * size / 2];
  }
};
}

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

/// \file edge_detector.cc
///
///             -------Example--------
///           Edge detector based on Sobel
///
/// \brief This example implements an edge detection based on Sobel mask
///

// include VisionCpp
#include <visioncpp.hpp>

#ifdef USE_CIMG
#include "CImg.h"
using namespace cimg_library;
#else
#include <opencv2/opencv.hpp>
#endif

// functor that calculates the magnitude of each pixel based on Sobel
struct OP_Magnitude {
  template <typename T1, typename T2>
  float operator()(const T1& t1, const T2& t2) {
    return cl::sycl::clamp(cl::sycl::sqrt(t1 * t1 + t2 * t2), 0.0f, 1.0f);
  }
};

void help() {
  std::cout << "Usage:" << std::endl;
  std::cout
      << "./bin/examples/edge_detector path/to/input.png path/to/output.png"
      << std::endl;
}

// main program
int main(int argc, char** argv) {
  if (argc != 3) {
    help();
    return -1;
  }
  // // selecting device
  auto dev = visioncpp::make_device<visioncpp::backend::sycl,
                                    visioncpp::device::cpu>();
  // defining the image size constants
  constexpr size_t COLS = 640;
  constexpr size_t ROWS = 640;
  constexpr size_t CHANNELS = 1;

  visioncpp::utils::IOHandler<COLS, ROWS, CHANNELS, unsigned char> ioHandler(
      argv[1]);

  // initializing the mask memories
  float sobel_x[9] = {-1.0f, 0.0f, 1.0f, -2.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f};
  float sobel_y[9] = {-1.0f, -2.0f, -1.0f, 0.0, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f};

  constexpr size_t filter_size = 3;
  constexpr size_t N = filter_size * filter_size;

  // creating a 3x3 filter mean filter
  float mean_array[N];
  for (size_t i = 0; i < N; i++) {
    mean_array[i] = 1.0f / static_cast<float>(N);
  }
  {
    //  the node which gets the input data
    auto in = visioncpp::terminal<visioncpp::pixel::U8C1, COLS, ROWS,
                                  visioncpp::memory_type::Buffer2D>(
        ioHandler.getInputPointer());

    // the node which gets the output data
    auto out = visioncpp::terminal<visioncpp::pixel::U8C1, COLS, ROWS,
                                   visioncpp::memory_type::Buffer2D>(
        ioHandler.getOutputPointer());

    // convert to Float
    auto fin = visioncpp::point_operation<visioncpp::OP_U8C1ToFloat>(in);

    // apply mean filter to smooth the image
    auto mean_filter =
        visioncpp::terminal<float, filter_size, filter_size,
                            visioncpp::memory_type::Buffer2D,
                            visioncpp::scope::Constant>(mean_array);
    auto mean = visioncpp::neighbour_operation<visioncpp::OP_Filter2D_One>(
        fin, mean_filter);

    // applying sobel_x filter
    auto x_filter =
        visioncpp::terminal<float, 3, 3, visioncpp::memory_type::Buffer2D,
                            visioncpp::scope::Constant>(sobel_x);
    auto sobel_x = visioncpp::neighbour_operation<visioncpp::OP_Filter2D_One>(
        mean, x_filter);

    auto y_filter =
        visioncpp::terminal<float, 3, 3, visioncpp::memory_type::Buffer2D,
                            visioncpp::scope::Constant>(sobel_y);
    auto sobel_y = visioncpp::neighbour_operation<visioncpp::OP_Filter2D_One>(
        mean, y_filter);

    auto intensity = visioncpp::point_operation<OP_Magnitude>(sobel_x, sobel_y);

    // convert to uchar
    auto uintensity =
        visioncpp::point_operation<visioncpp::OP_FloatToU8C1>(intensity);

    // assign operation
    auto pipe = visioncpp::assign(out, uintensity);

    visioncpp::execute<visioncpp::policy::Fuse, 8, 8, 8, 8>(pipe, dev);
  }

  ioHandler.save(argv[2]);

  return 0;
}

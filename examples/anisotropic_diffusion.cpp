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

/// \file anisotropic_diffusion.cc
///
///              ---------Example---------
///            Simplified Anisotropic Diffusion
///
/// \brief This example implements a simplified version of the Perona-Malik
/// Anisotropic Diffusion.
/// This technique removes noise from an image preserving the edges.
/// \param k - the edge preserving parameter [the smaller is the k, less things
///       are considered edges, the bigger, more things are considered as edges]
/// \param iters - number of iterations [it controls how blurry the image will
///        become, a higher number means a more blurry image]

// include OpenCV for camera display
#include <opencv2/opencv.hpp>

// include VisionCpp
#include <visioncpp.hpp>

// tunable parameters
constexpr float k{15.0f};    // edge preserving parameter
constexpr size_t iters{15};  // controls the blur

// operator which implements the simplified anisotropic diffusion
struct AniDiff {
  template <typename T>
  typename T::PixelType operator()(T nbr) {
    using Type = typename T::PixelType::data_type;

    // init output pixel
    cl::sycl::float4 out(0, 0, 0, 0);

    // init sum variable, which is used to normalize
    cl::sycl::float4 sum_w(0, 0, 0, 0);

    // get center pixel
    cl::sycl::float4 p1(nbr.at(nbr.I_c, nbr.I_r)[0],
                        nbr.at(nbr.I_c, nbr.I_r)[1],
                        nbr.at(nbr.I_c, nbr.I_r)[2], 0);

    // iterate over a 3x3 neighbourhood
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        // get neighbour pixel
        cl::sycl::float4 p2(nbr.at(nbr.I_c + i, nbr.I_r + j)[0],
                            nbr.at(nbr.I_c + i, nbr.I_r + j)[1],
                            nbr.at(nbr.I_c + i, nbr.I_r + j)[2], 0);

        // computes the weight which basically is the difference between pixels
        cl::sycl::float4 w = exp((-k) * cl::sycl::fabs(p1 - p2));

        // sum the weights for normalization
        sum_w += w;

        // store the output
        out += w * p2;
      }
    }
    // normalize output and return
    out = out / sum_w;
    return typename T::PixelType(static_cast<Type>(out.x()),
                                 static_cast<Type>(out.y()),
                                 static_cast<Type>(out.z()));
  }
};

// main program
int main(int argc, char **argv) {
  cv::VideoCapture cap(0);  // open the default camera
  if (!cap.isOpened()) {    // check if we succeeded
    std::cout << "Opening Camera Failed." << std::endl;
    return -1;
  }
  // selecting device using sycl as backend
  auto dev = visioncpp::make_device<visioncpp::backend::sycl,
                                    visioncpp::device::cpu>();

  // defining size constants
  constexpr size_t COLS = 640;
  constexpr size_t ROWS = 480;
  constexpr size_t CHNS = 3;

  // initialising output pointer
  std::shared_ptr<uchar> output(new uchar[COLS * ROWS * CHNS],
                                [](uchar *dataMem) { delete[] dataMem; });

  // initializing input and output image
  cv::Mat input;
  cv::Mat outImage(ROWS, COLS, CV_8UC(CHNS), output.get());

  /*
   This example contains a small expression tree
   but it uses a device memory which stores a temporary computation.
   So it is possible to create a loop during the computation
   and the data is always in the device. It just comes to the host in the last
   execute.

   Below is the expression tree used for this example

        (in_node)
         |
        (frgb)     [OP_U8C3ToF32C3] (convert uchar to float)
         |
   ---->(anidiff)  [AniDiff] (It iterates several times in the same node
   |     |                    applying the anisotropic diffusion serveral times)
   -------
         |
        (urgb)     [OP_F32C3ToU8C3] (convert float to uchar to display)


  */

  for (;;) {
    // Starting building the tree (use  {} during the creation of the tree)
    {
      // read frame
      cap.read(input);

      // resize image to the desirable size
      cv::resize(input, input, cv::Size(COLS, ROWS), 0, 0, cv::INTER_CUBIC);

      auto in_node =
          visioncpp::terminal<visioncpp::pixel::U8C3, COLS, ROWS,
                              visioncpp::memory_type::Buffer2D>(input.data);
      auto out_node =
          visioncpp::terminal<visioncpp::pixel::U8C3, COLS, ROWS,
                              visioncpp::memory_type::Buffer2D>(output.get());

      // device only memory
      auto device_memory =
          visioncpp::terminal<visioncpp::pixel::F32C3, COLS, ROWS,
                              visioncpp::memory_type::Buffer2D>();

      // convert to float
      auto frgb =
          visioncpp::point_operation<visioncpp::OP_U8C3ToF32C3>(in_node);

      // assign to temporary device memory
      auto exec1 = visioncpp::assign(device_memory, frgb);

      // apply anisotropic diffusion
      auto anidiff =
          visioncpp::neighbour_operation<AniDiff, 1, 1, 1, 1>(device_memory);

      // assign to the temporary device memory
      auto exec2 = visioncpp::assign(device_memory, anidiff);

      // convert to uchar
      auto urgb =
          visioncpp::point_operation<visioncpp::OP_F32C3ToU8C3>(device_memory);

      // assign to the host memory
      auto exec3 = visioncpp::assign(out_node, urgb);

      // execution (convert to float)
      visioncpp::execute<visioncpp::policy::Fuse, 32, 32, 16, 16>(exec1, dev);

      // apply anisotropic diffusion several times
      for (int i = 0; i < iters; i++) {
        visioncpp::execute<visioncpp::policy::Fuse, 32, 32, 16, 16>(exec2, dev);
      }

      // return image to host memory
      visioncpp::execute<visioncpp::policy::Fuse, 32, 32, 16, 16>(exec3, dev);
    }

    // display results
    cv::imshow("Reference Image", input);
    cv::imshow("Simplified Anisotropic Diffusion", outImage);

    // wait for key to finalize program
    if (cv::waitKey(1) >= 0) break;
  }
  return 0;
}

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

/// \file threhsold.cc
///
///              ---------Example---------
///                     Threshold
///
/// \brief This example implements a binary threshold
///
/// \param threshold - threshold value whose range is [0..1]
///

// include OpenCV for camera display
#include <opencv2/opencv.hpp>

// include VisionCpp
#include <visioncpp.hpp>

// main program
int main(int argc, char **argv) {
  // open video or camera
  cv::VideoCapture cap;

  if (argc == 1) {
    cap.open(0);
    std::cout << "To use video" << std::endl;
    std::cout << "example>: ./example path/to/video.avi" << std::endl;
  } else if (argc > 1) {
    cap.open(argv[1]);
  }

  // check if we succeeded
  if (!cap.isOpened()) {
    std::cout << "Opening Camera/Video Failed." << std::endl;
    return -1;
  }

  // selecting device using sycl as backend
  auto dev = visioncpp::make_device<visioncpp::backend::sycl,
                                    visioncpp::device::cpu>();

  // defining size constants
  constexpr size_t COLS = 640;
  constexpr size_t ROWS = 480;

  // initializing output pointer
  std::shared_ptr<uchar> output(new uchar[COLS * ROWS],
                                [](uchar *dataMem) { delete[] dataMem; });

  // initializing input and output image
  cv::Mat input;
  cv::Mat outImage(ROWS, COLS, CV_8UC1, output.get());

  // threshold parameter
  constexpr float threshold{0.5f};

  for (;;) {
    // Starting building the tree (use  {} during the creation of the tree)
    {
      // read frame
      cap.read(input);

      // check if image was loaded
      if (!input.data) {
        break;
      }

      // resize image to the desirable size
      cv::resize(input, input, cv::Size(COLS, ROWS), 0, 0, cv::INTER_CUBIC);

      auto in_node =
          visioncpp::terminal<visioncpp::pixel::U8C3, COLS, ROWS,
                              visioncpp::memory_type::Buffer2D>(input.data);
      auto out_node =
          visioncpp::terminal<visioncpp::pixel::U8C1, COLS, ROWS,
                              visioncpp::memory_type::Buffer2D>(output.get());

      // convert to Float
      auto frgb =
          visioncpp::point_operation<visioncpp::OP_U8C3ToF32C3>(in_node);

      // convert to grey
      auto fgrey = visioncpp::point_operation<visioncpp::OP_RGBToGREY>(frgb);

      // apply the threshold
      auto thresh_node =
          visioncpp::terminal<float, visioncpp::memory_type::Const>(
              static_cast<float>(threshold));
      auto thresh =
          visioncpp::point_operation<visioncpp::OP_Thresh>(fgrey, thresh_node);

      // scale threshold to display
      auto scale_node =
          visioncpp::terminal<float, visioncpp::memory_type::Const>(
              static_cast<float>(255.0f));
      auto urgb =
          visioncpp::point_operation<visioncpp::OP_Scale>(thresh, scale_node);

      // assign to the host memory
      auto k = visioncpp::assign(out_node, urgb);

      // execute
      visioncpp::execute<visioncpp::policy::Fuse, 32, 32, 16, 16>(k, dev);
    }

    // display results
    cv::imshow("Reference Image", input);
    cv::imshow("Threshold", outImage);

    // wait for key to finalize program
    if (cv::waitKey(1) >= 0) break;
  }

  // release video/camera
  cap.release();

  return 0;
}

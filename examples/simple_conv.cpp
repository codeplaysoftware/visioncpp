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

/// \file simple_conv.cc
///
///              ---------Example---------
///                 Simple Convolution
///
/// \brief This example implements a simple separable convolution using mean
/// filter
///

// include OpenCV for camera display
#include <opencv2/opencv.hpp>

// include  VisionCpp
#include <visioncpp.hpp>

// main program
int main() {
  // capture image via OpenCV
  cv::VideoCapture cap(0);  // open the default camera
  if (!cap.isOpened()) {    // check if we succeeded
    std::cout << "Opening Camera Failed." << std::endl;
    return -1;
  }
  // set fixed resolution on the camera
  constexpr size_t COLS = 640;
  constexpr size_t ROWS = 480;

  // where VisionCpp will run?
  auto dev = visioncpp::make_device<visioncpp::backend::sycl,
                                    visioncpp::device::cpu>();
  std::shared_ptr<unsigned char> vc_buffer(
      new unsigned char[COLS * ROWS * 3],
      [](unsigned char *dataMem) { delete[] dataMem; });

  // creating a 1x9 filter mean filter
  float filter_array[9] = {1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
                           1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f,
                           1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f};
  // create opencv mat
  cv::Mat frame;
  cv::Mat output;

  for (;;) {
    // read frame
    cap.read(frame);

    // resize image to the desirable size
    cv::resize(frame, frame, cv::Size(COLS, ROWS), 0, 0, cv::INTER_CUBIC);

    // column-wise filter for separable convolution
    auto filter_col =
        visioncpp::terminal<float, 9, 1, visioncpp::memory_type::Buffer2D,
                            visioncpp::scope::Constant>(filter_array);
    // row wise filter for separable convolution
    auto filter_row =
        visioncpp::terminal<float, 1, 9, visioncpp::memory_type::Buffer2D,
                            visioncpp::scope::Constant>(filter_array);
    // wrap OpenCV frame data in our leaf node
    auto data =
        visioncpp::terminal<visioncpp::pixel::U8C3, COLS, ROWS,
                            visioncpp::memory_type::Buffer2D>(frame.data);
    auto data_out =
        visioncpp::terminal<visioncpp::pixel::U8C3, COLS, ROWS,
                            visioncpp::memory_type::Buffer2D>(vc_buffer.get());

    auto node2 = visioncpp::neighbour_operation<visioncpp::OP_SepFilterCol>(
        data, filter_col);
    auto node3 = visioncpp::neighbour_operation<visioncpp::OP_SepFilterRow>(
        node2, filter_row);

    // writing back to Mat
    auto k = visioncpp::assign(data_out, node3);

    // execute the pipe
    visioncpp::execute<visioncpp::policy::Fuse, 32, 32, 16, 16>(k, dev);

    // show image
    output = cv::Mat(ROWS, COLS, CV_8UC3, vc_buffer.get());
    cv::imshow("Reference Image", frame);
    cv::imshow("Mean Filter Convolution", output);

    // loop
    if (cv::waitKey(30) >= 0) break;
  }
  return 0;
}

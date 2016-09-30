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

/// \file greyscale.cc
///
///             -------Example--------
///                  GreyScale
///
/// \brief this example capture the camera image and convert to grey scale
///

// include OpenCV for camera display
#include <opencv2/opencv.hpp>
// include VisionCpp
#include <visioncpp.hpp>

int main() {
  // capture image via OpenCV
  cv::VideoCapture cap(0);  // open the default camera
  if (!cap.isOpened()) {    // check if we succeeded
    std::cout << "Opening Camera Failed." << std::endl;
    return -1;
  }

  constexpr size_t COLS = 640;
  constexpr size_t ROWS = 480;

  // where VisionCpp will run.
  auto dev = visioncpp::make_device<visioncpp::backend::sycl,
                                    visioncpp::device::cpu>();

  // create a container for pipe output
  std::shared_ptr<unsigned char> img_cv(
      new unsigned char[COLS * ROWS],
      [](unsigned char *dataMem) { delete[] dataMem; });

  // create opencv mat
  cv::Mat frame;

  // point the img_cv to the Mat structure created for displaying purposes
  cv::Mat output(ROWS, COLS, CV_8UC1, img_cv.get());

  for (;;) {
    // read frame
    cap.read(frame);

    // resize image to the desirable size
    cv::resize(frame, frame, cv::Size(COLS, ROWS), 0, 0, cv::INTER_CUBIC);

    // create input node
    auto data =
        visioncpp::terminal<visioncpp::pixel::U8C3, COLS, ROWS,
                            visioncpp::memory_type::Buffer2D>(frame.data);
    // create output node
    auto data_out =
        visioncpp::terminal<visioncpp::pixel::U8C1, COLS, ROWS,
                            visioncpp::memory_type::Buffer2D>(img_cv.get());

    // unsigned char BGR to float RGB storage conversion
    auto node = visioncpp::point_operation<visioncpp::OP_CVBGRToRGB>(data);
    // convert RGB to GREY
    auto node2 = visioncpp::point_operation<visioncpp::OP_RGBToGREY>(node);
    // GREY is stored as a float so we need to do [0.0f, 1.0f] to [0, 255]
    // conversion
    auto node3 = visioncpp::point_operation<visioncpp::OP_GREYToCVBGR>(node2);

    // assign operation
    auto pipe = visioncpp::assign(data_out, node3);

    // execute the pipe
    visioncpp::execute<visioncpp::policy::Fuse, 8, 8, 8, 8>(pipe, dev);

    // show image
    cv::imshow("Reference Image", frame);
    cv::imshow("Greyscale", output);

    // esc?
    if (cv::waitKey(30) >= 0) break;
  }
  return 0;
}

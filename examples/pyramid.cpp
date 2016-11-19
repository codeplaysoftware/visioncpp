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

/// \file pyramid.cc
///
///             -------Example--------
///                Pyramid Example
///
/// \brief this example shows how pyramid node works.
///

// include OpenCV for camera display
#include <opencv2/opencv.hpp>
// include VisionCpp
#include <visioncpp.hpp>

using namespace visioncpp;
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

  constexpr size_t COLS = 640;
  constexpr size_t ROWS = 480;

  // where VisionCpp will run.
  auto dev = visioncpp::make_device<visioncpp::backend::sycl,
                                    visioncpp::device::cpu>();

  // create a container for downsampled ( lvl1 ) pyramid
  std::shared_ptr<unsigned char> img_vcpp_lv1(
      new unsigned char[COLS / 2 * ROWS / 2 * 3],
      [](unsigned char *dataMem) { delete[] dataMem; });

  std::shared_ptr<unsigned char> img_vcpp_lv2(
      new unsigned char[COLS / 4 * ROWS / 4],
      [](unsigned char *dataMem) { delete[] dataMem; });

  // filter for pyramid; change it to whatever you want
  float filter_array[3] = {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f};

  // create opencv mat
  cv::Mat frame;
  cv::Mat output_lvl1(ROWS / 2, COLS / 2, CV_8UC3, img_vcpp_lv1.get());
  cv::Mat output_lvl2(ROWS / 4, COLS / 4, CV_8UC1, img_vcpp_lv2.get());

  for (;;) {
    // read frame
    cap.read(frame);

    // check if image was loaded
    if (!frame.data) {
      break;
    }

    // resize image to the desirable size
    cv::resize(frame, frame, cv::Size(COLS, ROWS), 0, 0, cv::INTER_CUBIC);

    {
      // define the VisionCpp pipe
      // pyramid tree creation
      // column-wise filter for separable convolution in pyramid
      auto filter_col =
          terminal<float, 3, 1, memory_type::Buffer2D, scope::Constant>(
              filter_array);
      // row-wise filter for separable convolution in pyramid
      auto filter_row =
          terminal<float, 1, 3, memory_type::Buffer2D, scope::Constant>(
              filter_array);
      // define the VisionCpp pipe

      // input node ( terminal )
      auto data_in =
          terminal<visioncpp::pixel::U8C3, COLS, ROWS, memory_type::Buffer2D>(
              frame.data);

      auto pyr_node =
          pyramid_down<OP_SepFilterCol, OP_SepFilterRow, OP_DownsampleClosest,
                       2>(data_in, filter_col, filter_row);

      // does HSV on downsampled ( lvl 1 ) pyramid
      // create output node
      auto data_out_lvl1 =
          visioncpp::terminal<visioncpp::pixel::U8C3, COLS / 2, ROWS / 2,
                              visioncpp::memory_type::Buffer2D>(
              img_vcpp_lv1.get());

      // get pyramid ( U8C3 ) and convert it to F32C3
      auto node_hsv = visioncpp::point_operation<visioncpp::OP_U8C3ToF32C3>(
          pyr_node.get<0>());
      // convert RGB to HSV
      auto node2_hsv =
          visioncpp::point_operation<visioncpp::OP_RGBToHSV>(node_hsv);
      // F32C3 to U8C3 with order of BGR
      auto node3_hsv =
          visioncpp::point_operation<visioncpp::OP_HSVToU8C3>(node2_hsv);

      // assign operation
      auto hsv_node = visioncpp::assign(data_out_lvl1, node3_hsv);

      // lets do GREY on downsampled ( lvl 2 ) pyramid
      // create output node
      auto data_out_lvl2 =
          visioncpp::terminal<visioncpp::pixel::U8C1, COLS / 4, ROWS / 4,
                              visioncpp::memory_type::Buffer2D>(
              img_vcpp_lv2.get());

      // get pyramid ( U8C3 ) and convert it to F32C3
      auto node_grey = visioncpp::point_operation<visioncpp::OP_U8C3ToF32C3>(
          pyr_node.get<1>());
      // convert RGB to GREY
      auto node2_grey =
          visioncpp::point_operation<visioncpp::OP_RGBToGREY>(node_grey);

      auto node3_grey =
          visioncpp::point_operation<visioncpp::OP_GREYToCVBGR>(node2_grey);

      // assign operation
      auto grey_node = visioncpp::assign(data_out_lvl2, node3_grey);
      // execute the pipe
      // next the hsv

      visioncpp::execute<visioncpp::policy::Fuse, 32, 32, 16, 16>(hsv_node,
                                                                  dev);
      // next the grey
      visioncpp::execute<visioncpp::policy::Fuse, 32, 32, 16, 16>(grey_node,
                                                                  dev);
    }
    // show reference image
    cv::imshow("Reference Image", frame);

    // show pyramid level 1 image
    cv::imshow("Pyramid lvl 1: HSV", output_lvl1);

    // show pyramid level 2 image
    cv::imshow("Pyramid lvl 2: Greyscale", output_lvl2);

    // esc?
    if (cv::waitKey(1) >= 0) break;
  }

  // release video/camera
  cap.release();

  return 0;
}

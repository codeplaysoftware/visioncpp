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

/// \file depth_map_from_2_images.cpp
///
///              ---------Example---------
///            Depth Map From 2 images
///
/// \brief This example implements a Depth Map reconstruction from two images
/// using the block match algorithm
/// \param blockSize - this parameter defines the size of the block to be
/// searched
/// \param maxDisp - this parameter defines the maximum disparity between
/// pixels. It means the max number of pixels it will be used to search for the
/// best match.

// include OpenCV for camera display
#include <opencv2/opencv.hpp>

// include VisionCpp
#include <visioncpp.hpp>

// include limits to define infinity
#include <limits>

// Tunable parameters for the algorithm
constexpr int blockSize = 11;
constexpr int maxDisp = 25;

constexpr int halfBlock = blockSize / 2;

// Stereo block matching algorithm for depth map reconstruction
struct Stereo_BMA {
  // function that computers the sum of absolute difference of two blocks
  float SAD(const float im1[blockSize * blockSize],
            const float im2[blockSize * blockSize]) {
    float r = 0;
    for (size_t i = 0; i < blockSize * blockSize; i++) {
      r += cl::sycl::fabs(im1[i] - im2[i]);
    }
    return r;
  }

  // Function to get block of size blockSize frin (c,r) pixel
  template <typename T>
  void getBlock(const T& I, const int& c, const int& r, const int& layer,
                float block[blockSize * blockSize]) {
    int cnt = 0;
    for (int i2 = -halfBlock; i2 <= halfBlock; i2++) {
      for (int j2 = -halfBlock; j2 <= halfBlock; j2++) {
        block[cnt++] = I.at(c + i2, r + j2)[layer];
      }
    }
  }

  // function that computes the depth map
  template <typename T>
  visioncpp::pixel::U8C1 operator()(const T& I) {
    // get block from left image
    float block_l[blockSize * blockSize];
    getBlock(I, I.I_c, I.I_r, 0, block_l);

    // start with best sum of absolute difference equals to infinity
    float bestSAD = std::numeric_limits<float>::infinity();
    int bestJ = I.I_c;

    // for loop to find best match
    float block_r[blockSize * blockSize];
    for (int m = 0; m < maxDisp; m++) {
      // get block from right image
      getBlock(I, I.I_c - m, I.I_r, 1, block_r);

      // sum of absolute difference
      float temp = SAD(block_l, block_r);

      // store the smallest SAD
      if (temp < bestSAD) {
        bestSAD = temp;
        bestJ = I.I_c - m;
      }
    }

    // Compute disparity value
    return visioncpp::pixel::U8C1(I.I_c - bestJ);
  }
};

// main program
int main(int argc, char** argv) {
  // load images left and write in gray scale
  cv::Mat input_l =
      cv::imread("../data/tsukuba_l.png", CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat input_r =
      cv::imread("../data/tsukuba_r.png", CV_LOAD_IMAGE_GRAYSCALE);

  if (!input_l.data || !input_r.data) {  // check if we succeeded
    std::cout << "Loading image failed." << std::endl;
    return -1;
  }

  // selecting device
  auto dev = visioncpp::make_device<visioncpp::backend::sycl,
                                    visioncpp::device::cpu>();

  // defining the image size constants
  constexpr size_t COLS = 640;
  constexpr size_t ROWS = 480;

  // resize images
  cv::resize(input_l, input_l, cv::Size(COLS, ROWS), 0, 0, cv::INTER_CUBIC);
  cv::resize(input_r, input_r, cv::Size(COLS, ROWS), 0, 0, cv::INTER_CUBIC);

  // Shared Memory variable
  constexpr size_t SM = 16;

  // creating a pointer to store the results
  std::shared_ptr<unsigned char> output(
      new unsigned char[COLS * ROWS],
      [](unsigned char* dataMem) { delete[] dataMem; });

  // init output image from OpenCV for displaying results
  cv::Mat outputImage(ROWS, COLS, CV_8UC1, output.get());

  // the node which gets the input data from OpenCV
  {
    // store the left image in the terminal node
    auto in_l =
        visioncpp::terminal<visioncpp::pixel::U8C1, COLS, ROWS,
                            visioncpp::memory_type::Buffer2D>(input_l.data);

    // store the right image in the terminal node
    auto in_r =
        visioncpp::terminal<visioncpp::pixel::U8C1, COLS, ROWS,
                            visioncpp::memory_type::Buffer2D>(input_r.data);

    // the node which gets the output data
    auto out =
        visioncpp::terminal<visioncpp::pixel::U8C1, COLS, ROWS,
                            visioncpp::memory_type::Buffer2D>(output.get());

    // convert images to float
    auto fgrey_l = visioncpp::point_operation<visioncpp::OP_U8C1ToFloat>(in_l);
    auto fgrey_r = visioncpp::point_operation<visioncpp::OP_U8C1ToFloat>(in_r);

    // merge images
    auto merge =
        visioncpp::point_operation<visioncpp::OP_Merge2Chns>(fgrey_l, fgrey_r);

    // compute depth map
    auto depth = visioncpp::neighbour_operation<
        Stereo_BMA, halfBlock, halfBlock + maxDisp, halfBlock, halfBlock>(
        merge);

    // convert to unsigned char for displaying purposes
    // scale threhold to display
    auto scale_node = visioncpp::terminal<float, visioncpp::memory_type::Const>(
        static_cast<float>(8.0f));
    auto display =
        visioncpp::point_operation<visioncpp::OP_Scale>(depth, scale_node);

    // assign to the output
    auto exec = visioncpp::assign(out, display);

    // execute expression tree
    visioncpp::execute<visioncpp::policy::Fuse, SM, SM, SM, SM>(exec, dev);
  }

  // Display results
  cv::imshow("Image left", input_l);
  cv::imshow("Image right", input_r);
  cv::imshow("Depth Map", outputImage);

  // wait key to be pressed to finalize
  cv::waitKey(0);

  return 0;
}

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
///                    Bayer Filter
///
/// \brief This example implements the Bayer Filter Demosaic Method

// include OpenCV for camera display
#include <opencv2/opencv.hpp>

// include VisionCpp
#include <visioncpp.hpp>

// \brief functor which converts bayerRGGB to BGR
struct BayerRGGBToBGR {
  template <typename T>
  visioncpp::pixel::U8C3 operator()(T bayer) {
    // finding the pattern based on the index
    int _case = 0;
    if (bayer.I_r % 2 == 0 && bayer.I_c % 2 == 0) {
      _case = 1;
    } else if (bayer.I_r % 2 == 0 && bayer.I_c % 2 == 1) {
      _case = 2;
    } else if (bayer.I_r % 2 == 1 && bayer.I_c % 2 == 0) {
      _case = 3;
    } else {
      _case = 4;
    }
    switch (_case) {
      case 1:  // PIXEL R
      {
        // Init RGB variables
        uchar R;
        uchar G;
        uchar B;

        // Get G
        uchar G1 = bayer.at(bayer.I_c, bayer.I_r - 1)[0];
        uchar G2 = bayer.at(bayer.I_c + 1, bayer.I_r)[0];
        uchar G3 = bayer.at(bayer.I_c, bayer.I_r + 1)[0];
        uchar G4 = bayer.at(bayer.I_c - 1, bayer.I_r)[0];

        // Get R
        uchar R1 = bayer.at(bayer.I_c, bayer.I_r - 2)[0];
        uchar R2 = bayer.at(bayer.I_c + 2, bayer.I_r)[0];
        uchar R3 = bayer.at(bayer.I_c, bayer.I_r + 2)[0];
        uchar R4 = bayer.at(bayer.I_c - 2, bayer.I_r)[0];

        // Get B
        uchar B1 = bayer.at(bayer.I_c - 1, bayer.I_r - 1)[0];
        uchar B2 = bayer.at(bayer.I_c + 1, bayer.I_r - 1)[0];
        uchar B3 = bayer.at(bayer.I_c + 1, bayer.I_r + 1)[0];
        uchar B4 = bayer.at(bayer.I_c - 1, bayer.I_r + 1)[0];

        // Assign R
        R = bayer.at(bayer.I_c, bayer.I_r)[0];

        // Assign G
        if (abs(R1 - R3) < abs(R2 - R4)) {
          G = (G1 + G3) / 2;
        } else if (abs(R1 - R3) > abs(R2 - R4)) {
          G = (G2 + G4) / 2;
        } else {
          G = (G1 + G2 + G3 + G4) / 4;
        }

        // Assign B
        B = (B1 + B2 + B3 + B4) / 4;

        // Return BGR
        return visioncpp::pixel::U8C3(B, G, R);
      } break;
      case 2:  // PIXEL G1
      {
        // Init RGB variables
        uchar R;
        uchar G;
        uchar B;

        // Get R
        uchar R1 = bayer.at(bayer.I_c - 1, bayer.I_r)[0];
        uchar R2 = bayer.at(bayer.I_c + 1, bayer.I_r)[0];

        // Get B
        uchar B1 = bayer.at(bayer.I_c, bayer.I_r - 1)[0];
        uchar B2 = bayer.at(bayer.I_c, bayer.I_r + 1)[0];

        // Assign R
        R = (R1 + R2) / 2;

        // Assign G
        G = bayer.at(bayer.I_c, bayer.I_r)[0];

        // Assign B
        B = (B1 + B2) / 2;

        // Return BGR
        return visioncpp::pixel::U8C3(B, G, R);
      } break;
      case 3:  // Pixel G2
      {
        // Init RGB variables
        uchar R;
        uchar G;
        uchar B;

        // Get R
        uchar R1 = bayer.at(bayer.I_c, bayer.I_r - 1)[0];
        uchar R2 = bayer.at(bayer.I_c, bayer.I_r + 1)[0];

        // Get B
        uchar B1 = bayer.at(bayer.I_c - 1, bayer.I_r)[0];
        uchar B2 = bayer.at(bayer.I_c + 1, bayer.I_r)[0];

        // Assign R
        R = (R1 + R2) / 2;

        // Assign G
        G = bayer.at(bayer.I_c, bayer.I_r)[0];

        // Assign B
        B = (B1 + B2) / 2;

        // Return BGR
        return visioncpp::pixel::U8C3(B, G, R);
      } break;
      case 4:  // pixel B
      {
        // Init RGB Values
        uchar R;
        uchar G;
        uchar B;

        // Get G
        uchar G1 = bayer.at(bayer.I_c, bayer.I_r - 1)[0];
        uchar G2 = bayer.at(bayer.I_c + 1, bayer.I_r)[0];
        uchar G3 = bayer.at(bayer.I_c, bayer.I_r + 1)[0];
        uchar G4 = bayer.at(bayer.I_c - 1, bayer.I_r)[0];

        // Get B
        uchar B1 = bayer.at(bayer.I_c, bayer.I_r - 2)[0];
        uchar B2 = bayer.at(bayer.I_c + 2, bayer.I_r)[0];
        uchar B3 = bayer.at(bayer.I_c, bayer.I_r - 2)[0];
        uchar B4 = bayer.at(bayer.I_c - 2, bayer.I_r)[0];

        // Get R
        uchar R1 = bayer.at(bayer.I_c - 1, bayer.I_r - 1)[0];
        uchar R2 = bayer.at(bayer.I_c + 1, bayer.I_r - 1)[0];
        uchar R3 = bayer.at(bayer.I_c + 1, bayer.I_r + 1)[0];
        uchar R4 = bayer.at(bayer.I_c - 1, bayer.I_r + 1)[0];

        // Assign R
        R = (R1 + R2 + R3 + R4) / 4;

        // Assign G
        if (abs(B1 - B3) < abs(B2 - B4)) {
          G = (G1 + G3) / 2;
        } else if (abs(B1 - B3) > abs(B2 - B4)) {
          G = (G2 + G4) / 2;
        } else {
          G = (G1 + G2 + G3 + G4) / 4;
        }

        // Assign B
        B = bayer.at(bayer.I_c, bayer.I_r)[0];

        // Return BGR
        return visioncpp::pixel::U8C3(B, G, R);
      } break;
      default: {
        return visioncpp::pixel::U8C3(0, 0, 0);  // avoid warning
      } break;
    }
  }
};

// main program
int main(int argc, char **argv) {
  // Load the
  cv::Mat bayer = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

  if (!bayer.data) {
    std::cout << "Image not loaded, BayerRGGB image required as input"
              << std::endl;
    std::cout << "example>: ./bayer_filter raw_image.png" << std::endl;
    return -1;
  }

  constexpr size_t COLS = 1280;
  constexpr size_t ROWS = 720;

  // initializing input pointer
  std::shared_ptr<uchar> input_ptr(new uchar[COLS * ROWS],
                                   [](uchar *dataMem) { delete[] dataMem; });

  // initializing output pointer
  std::shared_ptr<uchar> output_ptr(new uchar[COLS * ROWS * 3],
                                    [](uchar *dataMem) { delete[] dataMem; });

  // selecting device using sycl as backend
  auto dev = visioncpp::make_device<visioncpp::backend::sycl,
                                    visioncpp::device::cpu>();

  cv::Mat outImage(ROWS, COLS, CV_8UC3, output_ptr.get());

  // Starting building the tree (use  {} during the creation of the tree)
  {
    // Init input node with the bayer image
    auto in_node =
        visioncpp::terminal<visioncpp::pixel::U8C1, COLS, ROWS,
                            visioncpp::memory_type::Buffer2D>(bayer.data);

    // Init output node
    auto out_node =
        visioncpp::terminal<visioncpp::pixel::U8C3, COLS, ROWS,
                            visioncpp::memory_type::Buffer2D>(output_ptr.get());

    // Apply demoisaic method (the 2,2,2,2 parameter means the Halo in the Top,
    // Left, Right, Bottom )
    auto bgr =
        visioncpp::neighbour_operation<BayerRGGBToBGR, 2, 2, 2, 2>(in_node);

    // assign to the host memory
    auto k = visioncpp::assign(out_node, bgr);

    // execute
    visioncpp::execute<visioncpp::policy::Fuse, 32, 32, 16, 16>(k, dev);
  }

  // display results
  cv::imshow("Reference Image", bayer);
  cv::imshow("Demosaic", outImage);

  // wait for key to finalize program
  cv::waitKey(0);

  return 0;
}

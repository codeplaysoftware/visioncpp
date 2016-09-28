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
/// \brief This example implements the Bayer Filter

// include OpenCV for camera display
#include <opencv2/opencv.hpp>

// include VisionCpp
#include <visioncpp.hpp>

struct BayerRGGBToBGR {
  template <typename T>
  visioncpp::pixel::U8C3 operator()(T bayer) {
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
        uchar R;
        uchar G;
        uchar B;

        uchar G1 = bayer.at(bayer.I_c, bayer.I_r - 1)[0];
        uchar G2 = bayer.at(bayer.I_c + 1, bayer.I_r)[0];
        uchar G3 = bayer.at(bayer.I_c, bayer.I_r + 1)[0];
        uchar G4 = bayer.at(bayer.I_c - 1, bayer.I_r)[0];
        uchar R1 = bayer.at(bayer.I_c, bayer.I_r - 2)[0];
        uchar R2 = bayer.at(bayer.I_c + 2, bayer.I_r)[0];
        uchar R3 = bayer.at(bayer.I_c, bayer.I_r + 2)[0];
        uchar R4 = bayer.at(bayer.I_c - 2, bayer.I_r)[0];
        uchar B1 = bayer.at(bayer.I_c - 1, bayer.I_r - 1)[0];
        uchar B2 = bayer.at(bayer.I_c + 1, bayer.I_r - 1)[0];
        uchar B3 = bayer.at(bayer.I_c + 1, bayer.I_r + 1)[0];
        uchar B4 = bayer.at(bayer.I_c - 1, bayer.I_r + 1)[0];

        // R
        R = bayer.at(bayer.I_c, bayer.I_r)[0];

        // G
        if (abs(R1 - R3) < abs(R2 - R4)) {
          G = (G1 + G3) / 2;
        } else if (abs(R1 - R3) > abs(R2 - R4)) {
          G = (G2 + G4) / 2;
        } else {
          G = (G1 + G2 + G3 + G4) / 4;
        }

        // B
        B = (B1 + B2 + B3 + B4) / 4;

        return visioncpp::pixel::U8C3(B, G, R);
      } break;
      case 2:  // PIXEL G1
      {
        uchar R;
        uchar G;
        uchar B;
        uchar R1 = bayer.at(bayer.I_c - 1, bayer.I_r)[0];
        uchar R2 = bayer.at(bayer.I_c + 1, bayer.I_r)[0];
        uchar B1 = bayer.at(bayer.I_c, bayer.I_r - 1)[0];
        uchar B2 = bayer.at(bayer.I_c, bayer.I_r + 1)[0];

        // R
        R = (R1 + R2) / 2;

        // G
        G = bayer.at(bayer.I_c, bayer.I_r)[0];

        // B
        B = (B1 + B2) / 2;

        return visioncpp::pixel::U8C3(B, G, R);
      } break;
      case 3:
        // Pixel G2
        {
          uchar R;
          uchar G;
          uchar B;
          uchar R1 = bayer.at(bayer.I_c, bayer.I_r - 1)[0];
          uchar R2 = bayer.at(bayer.I_c, bayer.I_r + 1)[0];
          uchar B1 = bayer.at(bayer.I_c - 1, bayer.I_r)[0];
          uchar B2 = bayer.at(bayer.I_c + 1, bayer.I_r)[0];

          // R
          R = (R1 + R2) / 2;

          // G
          G = bayer.at(bayer.I_c, bayer.I_r)[0];

          // B
          B = (B1 + B2) / 2;

          return visioncpp::pixel::U8C3(B, G, R);
        }
        break;
      case 4:  // pixel B
      {
        uchar R;
        uchar G;
        uchar B;

        uchar G1 = bayer.at(bayer.I_c, bayer.I_r - 1)[0];
        uchar G2 = bayer.at(bayer.I_c + 1, bayer.I_r)[0];
        uchar G3 = bayer.at(bayer.I_c, bayer.I_r + 1)[0];
        uchar G4 = bayer.at(bayer.I_c - 1, bayer.I_r)[0];
        uchar B1 = bayer.at(bayer.I_c, bayer.I_r - 2)[0];
        uchar B2 = bayer.at(bayer.I_c + 2, bayer.I_r)[0];
        uchar B3 = bayer.at(bayer.I_c, bayer.I_r - 2)[0];
        uchar B4 = bayer.at(bayer.I_c - 2, bayer.I_r)[0];
        uchar R1 = bayer.at(bayer.I_c - 1, bayer.I_r - 1)[0];
        uchar R2 = bayer.at(bayer.I_c + 1, bayer.I_r - 1)[0];
        uchar R3 = bayer.at(bayer.I_c + 1, bayer.I_r + 1)[0];
        uchar R4 = bayer.at(bayer.I_c - 1, bayer.I_r + 1)[0];

        // R
        R = (R1 + R2 + R3 + R4) / 4;

        // G
        if (abs(B1 - B3) < abs(B2 - B4)) {
          G = (G1 + G3) / 2;
        } else if (abs(B1 - B3) > abs(B2 - B4)) {
          G = (G2 + G4) / 2;
        } else {
          G = (G1 + G2 + G3 + G4) / 4;
        }

        // B
        B = bayer.at(bayer.I_c, bayer.I_r)[0];

        return visioncpp::pixel::U8C3(B, G, R);
      } break;
    }
    return visioncpp::pixel::U8C3(0, 0, 0);
  }
};

// main program
int main(int argc, char **argv) {
  cv::Mat im = cv::imread("../media/lena.png");

  constexpr size_t cols_orig = 512;
  constexpr size_t rows_orig = 512;

  cv::resize(im, im, cv::Size(cols_orig, rows_orig));

  // defining size constants
  constexpr size_t COLS = cols_orig * 2;
  constexpr size_t ROWS = rows_orig * 2;

  // initializing input pointer
  std::shared_ptr<uchar> input_ptr(new uchar[COLS * ROWS],
                                   [](uchar *dataMem) { delete[] dataMem; });

  // initializing output pointer
  std::shared_ptr<uchar> output_ptr(new uchar[COLS * ROWS * 3],
                                    [](uchar *dataMem) { delete[] dataMem; });

  // creating bayer patter from input image
  cv::Mat bayer(COLS, ROWS, CV_8UC1, input_ptr.get());
  for (size_t ib = 0, ii = 0; ib < bayer.rows; ib += 2, ii++) {
    for (size_t jb = 0, jj = 0; jb < bayer.cols; jb += 2, jj++) {
      // std::cout << ib << " " << jb << std::endl;
      // std::cout << ii << " " << jj << std::endl;
      bayer.at<uchar>(ib, jb) = im.at<cv::Vec3b>(ii, jj)[2];
      bayer.at<uchar>(ib + 1, jb) = im.at<cv::Vec3b>(ii, jj)[1];
      bayer.at<uchar>(ib, jb + 1) = im.at<cv::Vec3b>(ii, jj)[1];
      bayer.at<uchar>(ib + 1, jb + 1) = im.at<cv::Vec3b>(ii, jj)[0];
    }
  }

  // selecting device using sycl as backend
  auto dev = visioncpp::make_device<visioncpp::backend::sycl,
                                    visioncpp::device::cpu>();

  cv::Mat outImage(ROWS, COLS, CV_8UC3, output_ptr.get());

  // Starting building the tree (use  {} during the creation of the tree)
  {
    auto in_node =
        visioncpp::terminal<visioncpp::pixel::U8C1, COLS, ROWS,
                            visioncpp::memory_type::Buffer2D>(bayer.data);
    auto out_node =
        visioncpp::terminal<visioncpp::pixel::U8C3, COLS, ROWS,
                            visioncpp::memory_type::Buffer2D>(output_ptr.get());

    auto bayer =
        visioncpp::neighbour_operation<BayerRGGBToBGR, 2, 2, 2, 2>(in_node);

    // assign to the host memory
    auto k = visioncpp::assign(out_node, bayer);

    // execute
    visioncpp::execute<visioncpp::policy::Fuse, 32, 32, 16, 16>(k, dev);
  }

  // display results
  cv::imshow("Reference Image", bayer);
  cv::imshow("Demoisaic", outImage);

  // wait for key to finalize program

  cv::waitKey(0);

  return 0;
}

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

/// \file optical_flow_LK.cc
///
///             -------Example--------
///             Lucas-Kanade Optical Flow
///
/// \brief This example implements the Lucas-Kanade Optical Flow
/// This optical flow needs to solve the following equation
/// | u | = inv(sum w  | Dx^2  Dx*Dy | ) * sum w | -Dx*Dt |
/// | v |      (       | Dx*Dy Dy^2  | )         | -Dy*Dt |
/// where w is the 3x3 window of ones
///
/// doing algebraic manipulation
/// n = sum(Dx^2) * sum (Dy^2) - (sum (Dx * Dy)^2)
/// u = (-sum (Dy^2) * sum (Dx*Dt) + sum (Dx*Dy) * sum (Dy*Dt)) / n
/// v = ( sum (Dx*Dt) * sum (Dx*Dy) - sum (Dx^2) * sum (Dy*Dt)) / n

// include OpenCV for camera display
#include <opencv2/opencv.hpp>

// include VisionCPP
#include <visioncpp.hpp>

// operator that transforms uv coordinates into polar coordinates
// it was created for displaying the optical flow in RGB
struct OP_UVtoPolar {
  visioncpp::pixel::F32C3 operator()(visioncpp::pixel::F32C2 t) {
    float intensity = cl::sycl::sqrt(t[0] * t[0] + t[1] * t[1]) / 2.0f;
    float angle = cl::sycl::atan2(t[1], t[0]) / (2.0f * M_PI);
    float chn = 1.0f;
    return visioncpp::pixel::F32C3(angle, chn, intensity);
  }
};

// function to convert the (u,v) matrix to colors for displaying purposes
template <size_t COLS, size_t ROWS, size_t SM, typename Device>
void displayOpticalFlow(const std::shared_ptr<float> uv,
                        std::shared_ptr<uchar> rgbFlow, Device &dev) {
  // output node that will store the color information
  auto out =
      visioncpp::terminal<visioncpp::pixel::U8C3, COLS, ROWS,
                          visioncpp::memory_type::Buffer2D>(rgbFlow.get());

  // input UV
  auto inUV = visioncpp::terminal<visioncpp::pixel::F32C2, COLS, ROWS,
                                  visioncpp::memory_type::Buffer2D>(uv.get());
  // convert UV into polar coordinates
  auto polar = visioncpp::point_operation<OP_UVtoPolar>(inUV);

  // convert into RGB
  auto frgb = visioncpp::point_operation<visioncpp::OP_HSVToRGB>(polar);

  // convert float to char
  auto urgb = visioncpp::point_operation<visioncpp::OP_F32C3ToU8C3>(frgb);

  // assign the urgb to the output node
  auto k = visioncpp::assign(out, urgb);

  // execute
  visioncpp::execute<visioncpp::policy::Fuse, SM, SM, SM, SM>(k, dev);
}

// main program
int main(int argc, char **argv) {
  // open camera using OpenCV
  cv::VideoCapture cap(0);  // open the default camera
  if (!cap.isOpened())      // check if we succeeded
    return -1;

  // selecting device using sycl as backend
  auto dev = visioncpp::make_device<visioncpp::backend::sycl,
                                    visioncpp::device::gpu>();

  // defining the image size constants
  constexpr size_t COLS = 640;
  constexpr size_t ROWS = 480;
  constexpr size_t SM = 16;

  // initializing the mask memories
  float sum_mask[9] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  float prewitt_x[9] = {-1.0f, 0.0f,  1.0f, -2.0f, 0.0f,
                        2.0f,  -1.0f, 0.0f, 1.0f};
  float prewitt_y[9] = {-1.0f, -2.0f, -1.0f, 0.0, 0.0f, 0.0f, 1.0f, 2.0f, 1.0f};

  // initializing pointers which will store the final results
  std::shared_ptr<float> outputUV(new float[COLS * ROWS * 2],
                                  [](float *dataMem) { delete[] dataMem; });
  std::shared_ptr<uchar> rgbflow(new uchar[COLS * ROWS * 3],
                                 [](uchar *dataMem) { delete[] dataMem; });

  // initializing matrices which will store current and previous frame
  cv::Mat current;
  cv::Mat previous;

  // init matrix which displays in RGB the (u,v) matrix
  cv::Mat rgbflow_mat(ROWS, COLS, CV_8UC3, rgbflow.get());

  // flag to control if it is the first frame
  bool first = true;

  for (;;) {
    // don't capture the first frame.
    // it needs the previous frame to compute optical flow
    if (first) {
      // read frame
      cap.read(current);

      // resize image to the desirable size
      cv::resize(current, current, cv::Size(COLS, ROWS), 0, 0, cv::INTER_CUBIC);

      // set flag to false
      first = false;

      continue;
    }

    // Starting building the tree (use  {} during the creation of the tree)
    {
      // copy previous frame
      previous = current.clone();

      // read frame
      cap.read(current);

      // resize image to the desirable size
      cv::resize(current, current, cv::Size(COLS, ROWS), 0, 0, cv::INTER_CUBIC);

      // init terminal nodes
      auto outUV =
          visioncpp::terminal<visioncpp::pixel::F32C2, COLS, ROWS,
                              visioncpp::memory_type::Buffer2D>(outputUV.get());

      auto in =
          visioncpp::terminal<visioncpp::pixel::U8C3, COLS, ROWS,
                              visioncpp::memory_type::Buffer2D>(current.data);
      auto prev =
          visioncpp::terminal<visioncpp::pixel::U8C3, COLS, ROWS,
                              visioncpp::memory_type::Buffer2D>(previous.data);

      // convert unsigned char to float
      auto ifrgb = visioncpp::point_operation<visioncpp::OP_U8C3ToF32C3>(in);
      // convert to grey scale previous and current frame
      auto ifgrey = visioncpp::point_operation<visioncpp::OP_RGBToGREY>(ifrgb);

      // convert unsigned char to float
      auto pfrgb = visioncpp::point_operation<visioncpp::OP_U8C3ToF32C3>(prev);
      // convert to grey scale previous and current frame
      auto pfgrey = visioncpp::point_operation<visioncpp::OP_RGBToGREY>(pfrgb);

      // apply derivatives in x, y and t (in current and previous images)
      auto px_filter =
          visioncpp::terminal<float, 3, 3, visioncpp::memory_type::Buffer2D,
                              visioncpp::scope::Constant>(prewitt_x);
      auto py_filter =
          visioncpp::terminal<float, 3, 3, visioncpp::memory_type::Buffer2D,
                              visioncpp::scope::Constant>(prewitt_y);

      // break tree before convolution
      auto iifgrey =
          visioncpp::schedule<visioncpp::policy::Fuse, SM, SM, SM, SM>(ifgrey);
      auto ppfgrey =
          visioncpp::schedule<visioncpp::policy::Fuse, SM, SM, SM, SM>(pfgrey);

      auto ipx = visioncpp::neighbour_operation<visioncpp::OP_Filter2D>(
          iifgrey, px_filter);
      auto ipy = visioncpp::neighbour_operation<visioncpp::OP_Filter2D>(
          iifgrey, py_filter);
      auto ppx = visioncpp::neighbour_operation<visioncpp::OP_Filter2D>(
          ppfgrey, px_filter);
      auto ppy = visioncpp::neighbour_operation<visioncpp::OP_Filter2D>(
          ppfgrey, py_filter);

      // break tree after convolution
      auto iipx =
          visioncpp::schedule<visioncpp::policy::Fuse, SM, SM, SM, SM>(ipx);
      auto iipy =
          visioncpp::schedule<visioncpp::policy::Fuse, SM, SM, SM, SM>(ipy);
      auto pppx =
          visioncpp::schedule<visioncpp::policy::Fuse, SM, SM, SM, SM>(ppx);
      auto pppy =
          visioncpp::schedule<visioncpp::policy::Fuse, SM, SM, SM, SM>(ppy);

      // sum derivatives of current and previous frames
      auto px = visioncpp::point_operation<visioncpp::OP_Add>(iipx, pppx);
      auto py = visioncpp::point_operation<visioncpp::OP_Add>(iipy, pppy);
      auto pt = visioncpp::point_operation<visioncpp::OP_Sub>(ifgrey, pfgrey);

      auto px2 = visioncpp::point_operation<visioncpp::OP_PowerOf2>(px);
      auto py2 = visioncpp::point_operation<visioncpp::OP_PowerOf2>(py);
      auto pxy = visioncpp::point_operation<visioncpp::OP_Mul>(px, py);
      auto pxt = visioncpp::point_operation<visioncpp::OP_Mul>(px, pt);
      auto pyt = visioncpp::point_operation<visioncpp::OP_Mul>(py, pt);

      // break tree before convolution
      auto ppx2 =
          visioncpp::schedule<visioncpp::policy::Fuse, SM, SM, SM, SM>(px2);
      auto ppy2 =
          visioncpp::schedule<visioncpp::policy::Fuse, SM, SM, SM, SM>(py2);
      auto ppxy =
          visioncpp::schedule<visioncpp::policy::Fuse, SM, SM, SM, SM>(pxy);
      auto ppxt =
          visioncpp::schedule<visioncpp::policy::Fuse, SM, SM, SM, SM>(pxt);
      auto ppyt =
          visioncpp::schedule<visioncpp::policy::Fuse, SM, SM, SM, SM>(pyt);

      // Sum neighbours
      auto sum_mask_node =
          visioncpp::terminal<float, 3, 3, visioncpp::memory_type::Buffer2D,
                              visioncpp::scope::Constant>(sum_mask);

      auto sumpx2 = visioncpp::neighbour_operation<visioncpp::OP_Filter2D>(
          ppx2, sum_mask_node);
      auto sumpy2 = visioncpp::neighbour_operation<visioncpp::OP_Filter2D>(
          ppy2, sum_mask_node);
      auto sumpxy = visioncpp::neighbour_operation<visioncpp::OP_Filter2D>(
          ppxy, sum_mask_node);
      auto sumpxt = visioncpp::neighbour_operation<visioncpp::OP_Filter2D>(
          ppxt, sum_mask_node);
      auto sumpyt = visioncpp::neighbour_operation<visioncpp::OP_Filter2D>(
          ppyt, sum_mask_node);

      // break tree after convolution
      auto ksumpx2 =
          visioncpp::schedule<visioncpp::policy::Fuse, SM, SM, SM, SM>(sumpx2);
      auto ksumpy2 =
          visioncpp::schedule<visioncpp::policy::Fuse, SM, SM, SM, SM>(sumpy2);
      auto ksumpxy =
          visioncpp::schedule<visioncpp::policy::Fuse, SM, SM, SM, SM>(sumpxy);
      auto ksumpxt =
          visioncpp::schedule<visioncpp::policy::Fuse, SM, SM, SM, SM>(sumpxt);
      auto ksumpyt =
          visioncpp::schedule<visioncpp::policy::Fuse, SM, SM, SM, SM>(sumpyt);

      // now we can finally apply the formulas
      // u = (-sum (Dy^2) * sum (Dx*Dt) + sum (Dx*Dy) * sum (Dy*Dt)) / n
      // v = ( sum (Dx*Dt) * sum (Dx*Dy) - sum (Dx^2) * sum (Dy*Dt)) / n

      // n = sum(Dx^2) * sum (Dy^2) - (sum (Dx * Dy)^2)
      auto px2py2 =
          visioncpp::point_operation<visioncpp::OP_Mul>(ksumpx2, ksumpy2);
      auto pxy2 = visioncpp::point_operation<visioncpp::OP_PowerOf2>(ksumpxy);
      auto px2py2_Sub_pxy2 =
          visioncpp::point_operation<visioncpp::OP_Sub>(px2py2, pxy2);
      auto norm = visioncpp::schedule<visioncpp::policy::Fuse, SM, SM, SM, SM>(
          px2py2_Sub_pxy2);

      // calculate V
      // v = ( sum (Dx*Dt) * sum (Dx*Dy) - sum (Dx^2) * sum (Dy*Dt)) / n
      auto pxtpxy =
          visioncpp::point_operation<visioncpp::OP_Mul>(ksumpxt, ksumpxy);
      auto px2pyt =
          visioncpp::point_operation<visioncpp::OP_Mul>(ksumpx2, ksumpyt);
      auto pxtpxy_Sub_px2pyt =
          visioncpp::point_operation<visioncpp::OP_Sub>(pxtpxy, px2pyt);
      auto kpxtpxy_Sub_px2pyt =
          visioncpp::schedule<visioncpp::policy::Fuse, SM, SM, SM, SM>(
              pxtpxy_Sub_px2pyt);

      auto v = visioncpp::point_operation<visioncpp::OP_Div>(kpxtpxy_Sub_px2pyt,
                                                             norm);

      // calculate U
      // u = (-sum (Dy^2) * sum (Dx*Dt) + sum (Dx*Dy) * sum (Dy*Dt)) / n
      auto pxypyt =
          visioncpp::point_operation<visioncpp::OP_Mul>(ksumpxy, ksumpyt);
      auto py2pxt =
          visioncpp::point_operation<visioncpp::OP_Mul>(ksumpy2, ksumpxt);
      auto pxypft_Sub_py2pxt =
          visioncpp::point_operation<visioncpp::OP_Sub>(pxypyt, py2pxt);
      auto kpxypft_Sub_py2pxt =
          visioncpp::schedule<visioncpp::policy::Fuse, SM, SM, SM, SM>(
              pxypft_Sub_py2pxt);

      auto u = visioncpp::point_operation<visioncpp::OP_Div>(kpxypft_Sub_py2pxt,
                                                             norm);

      // assign result in one matrix with 2 channels
      auto uv = visioncpp::point_operation<visioncpp::OP_Merge2Chns>(u, v);

      auto kuv = visioncpp::assign(outUV, uv);

      // execute expression tree
      visioncpp::execute<visioncpp::policy::Fuse, SM, SM, SM, SM>(kuv, dev);

      // convert (u,v) matrix into colors for better visualization
      displayOpticalFlow<COLS, ROWS, SM>(outputUV, rgbflow, dev);
    }
    // display optical flow
    cv::imshow("Reference Image", current);
    cv::imshow("Optical Flow Lucas-Kanade", rgbflow_mat);

    // check button pressed to exit the program
    if (cv::waitKey(1) >= 0) break;
  }

  return 0;
}

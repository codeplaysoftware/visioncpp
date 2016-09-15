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

/// \file rgb2hsv.cc
///
///             -------Example--------
///                  RGB -> HSV
/// \brief this example convert an RGB pixel to HSV

// include VisionCpp
#include <visioncpp.hpp>

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout << "You need to provide 3 uchar values" << std::endl;
    std::cout << "example>: ./example_rgb2hsv 100 13 145" << std::endl;
    return -1;
  }

  // where VisionCpp will run.
  auto dev = visioncpp::make_device<visioncpp::backend::sycl,
                                    visioncpp::device::cpu>();

  // create a host container for input data
  std::shared_ptr<unsigned char> in_rgb(
      new unsigned char[3], [](unsigned char *dataMem) { delete[] dataMem; });

  in_rgb.get()[0] = atoi(argv[1]);
  in_rgb.get()[1] = atoi(argv[2]);
  in_rgb.get()[2] = atoi(argv[3]);

  // create a host container for output data
  std::shared_ptr<unsigned char> out_hsv(
      new unsigned char[3], [](unsigned char *dataMem) { delete[] dataMem; });

  // exiting this scope will sync data
  {
    // definition of the VisionCpp pipeline:

    // create terminal nodes - a leaf node ( data node ) of the expression tree.
    // terminal struct takes 4 arguments
    // 1st template parameter specifies the data U8 (unsigned char) C3 (three
    // channels)
    // 2nd: the number of columns in the storage
    // 3rd: the number of rows in the storage
    // 4th: the underlying storage type - currently only Buffer2D supported
    auto data =
        visioncpp::terminal<visioncpp::pixel::U8C3, 1, 1,
                            visioncpp::memory_type::Buffer2D>(in_rgb.get());
    auto data_out =
        visioncpp::terminal<visioncpp::pixel::U8C3, 1, 1,
                            visioncpp::memory_type::Buffer2D>(out_hsv.get());

    // unsigned char -> float RGB storage conversion
    auto node = visioncpp::point_operation<visioncpp::OP_U8C3ToF32C3>(data);
    // float RGB to float HSV conversion
    auto node2 = visioncpp::point_operation<visioncpp::OP_RGBToHSV>(node);
    // helper node that allows display of HSV
    // for unsigned char: V <- 255*V, S <- 255*S, H <- H/2 ( to fit in range of
    // 0..255 )
    auto node3 = visioncpp::point_operation<visioncpp::OP_HSVToU8C3>(node2);

    // assign operation that writes output of the pipe to output terminal node
    auto pipe = visioncpp::assign(data_out, node3);
    // execute the pipeline
    // 1st template parameter defines if VisionCpp back-end fuses the expression
    // 2nd & 3rd shared memory sizes ( column, row )
    // 4th & 5th local work group size ( column , row )
    visioncpp::execute<visioncpp::policy::Fuse, 1, 1, 1, 1>(pipe, dev);
  }

  printf("RGB: %u %u %u \nHSV: %u %u %u \n", in_rgb.get()[0], in_rgb.get()[1],
         in_rgb.get()[2], out_hsv.get()[0], out_hsv.get()[1], out_hsv.get()[2]);

  return 0;
}

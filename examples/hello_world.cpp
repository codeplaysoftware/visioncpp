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

/// \file hello_world.cc
///
///             -------Example--------
///                  Hello World
///
/// \brief this example converts RGB pixel to Grey
///

// include VisionCpp library
#include <visioncpp.hpp>

/// \brief This functor performs conversion from [0, 255] to [0.0f, 1.0f]
struct MyNormaliseFunctor {
  /// \param in - three channel unsigned char
  /// \return F32C3 - three channel float
  visioncpp::pixel::F32C3 operator()(visioncpp::pixel::U8C3 in) {
    const float FLOAT_TO_BYTE = 255.0f;
    const float BYTE_TO_FLOAT = 1.0f / FLOAT_TO_BYTE;
    return visioncpp::pixel::F32C3(static_cast<float>(in[0] * BYTE_TO_FLOAT),
                                   static_cast<float>(in[1] * BYTE_TO_FLOAT),
                                   static_cast<float>(in[2] * BYTE_TO_FLOAT));
  }
};

/// \brief This functor performs RGB to grey convertion following rule:
/// GREY <- 0.299f * R + 0,587f * G + 0.114 * B
struct MyGreyFunctor {
  /// \param in - RGB pixel.
  /// \returns float - greyscale value.
  float operator()(visioncpp::pixel::F32C3 in) {
    return 0.299f * in[0] + 0.587f * in[1] + 0.114f * in[2];
  }
};

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cout << "You need to provide 3 uchar values" << std::endl;
    std::cout << "example>: ./example_hello_world 100 13 145" << std::endl;
    return -1;
  }

  // where VisionCpp will run.
  auto dev = visioncpp::make_device<visioncpp::backend::sycl,
                                    visioncpp::device::cpu>();

  // an image pixel to be converted to grey
  std::shared_ptr<unsigned char> in_rgb(new unsigned char[3]);
  in_rgb.get()[0] = atoi(argv[1]);
  in_rgb.get()[1] = atoi(argv[2]);
  in_rgb.get()[2] = atoi(argv[3]);

  // create a host container for output data
  std::shared_ptr<float> out_grey(new float[1]);

  // definition of the VisionCpp pipe:
  {
    // put expression tree in a curly braces in order to define the scope for
    // the expression
    // create terminal nodes - a leaf node in the expression tree.
    // terminal struct takes 4 arguments
    // 1st template parameter specifies the data U8 (unsigned char) C3 (three
    // channels)
    // 2nd number of columns in the storage
    // 3rd number of rows in the storage
    // 4th underlying storage type - currently only Buffer2D supported
    auto data =
        visioncpp::terminal<visioncpp::pixel::U8C3, 1, 1,
                            visioncpp::memory_type::Buffer2D>(in_rgb.get());
    auto data_out =
        visioncpp::terminal<visioncpp::pixel::F32C1, 1, 1,
                            visioncpp::memory_type::Buffer2D>(out_grey.get());

    // converting unsigned char value to normalised float value
    auto node = visioncpp::point_operation<MyNormaliseFunctor>(data);
    // float RGB to float Grey conversion
    auto node2 = visioncpp::point_operation<MyGreyFunctor>(node);
    // converting float value for the grey to the unsigned char
    // assign operation that writes output of the pipe to output terminal node
    auto pipe = visioncpp::assign(data_out, node2);

    // execute the pipe defined pipe
    // 1st template parameter defines if VisionCpp back-end fuses the expression
    // 2nd & 3rd shared memory sizes ( column, row )
    // 4th & 5th local work group size ( column , row )
    visioncpp::execute<visioncpp::policy::Fuse, 1, 1, 1, 1>(pipe, dev);
  }  // destroy the expression tree to bring the data back to host

  printf("RGB: %u %u %u \nGrey: %f \n", in_rgb.get()[0], in_rgb.get()[1],
         in_rgb.get()[2], out_grey.get()[0]);
}

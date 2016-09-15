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

#ifndef VISIONCPP_TESTS_INCLUDE_COMMON_HPP_
#define VISIONCPP_TESTS_INCLUDE_COMMON_HPP_

// opencv dep
// make sure that opencv is installed on the testing machine
#include <gtest/gtest.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <visioncpp.hpp>
#include <opencv/cv.h>

// used to generate test names
#define GLUE_HELPER(x, y) x##_##y
#define GLUE(x, y) GLUE_HELPER(x, y)
#define REAL_STRINGIZE(x) #x
#define STRINGIZE(x) REAL_STRINGIZE(x)

// verification function that takes OpenCVs Mat as a reference and
// and shared pointer that points to host storage with output from VisionCpps
// pipe.
template <typename T>
void verify(const cv::Mat &ref, std::shared_ptr<T> img) {
  // opencv channels
  int cv_cn = ref.channels();

  uint8_t *pixelPtr = (uint8_t *)ref.data;

  for (int i = 0; i < ref.rows; i++) {
    for (int j = 0; j < ref.cols; j++) {
      for (int c = 0; c < cv_cn; c++) {
        auto expected = (float)(pixelPtr[i * ref.cols * cv_cn + j * cv_cn + c]);
        auto tested = (float)(img.get()[i * ref.cols * cv_cn + j * cv_cn + c]);
        ASSERT_NEAR(expected, tested, 6)
            << "\nrow: " << i << " col: " << j << " channel: " << c
            << " expected: " << expected << " tested: " << tested;
      }
    }
  }
}

// singleton that is used for generating a data for tests
// create 256 textures that are 256x256 with all possible combinations of pixel
// values for unsigned char storage.
namespace common {
namespace singleton {
struct DataSet {
  std::vector<std::shared_ptr<unsigned char>> m_data;
  static const size_t m_width = 256;
  static const size_t m_height = 256;
  static const size_t m_depth = 256;

 public:
  static DataSet &Instance() {
    static DataSet m_dataset;
    return m_dataset;
  }

  DataSet(DataSet const &) = delete;
  DataSet(DataSet &&) = delete;
  DataSet &operator=(DataSet const &) = delete;
  DataSet &operator=(DataSet &&) = delete;

 protected:
  DataSet() {
    for (size_t b = 0; b < m_depth; b++) {
      int i = 0;
      auto element = std::shared_ptr<unsigned char>(
          new unsigned char[m_width * m_height * 3]);
      for (size_t g = 0; g < m_height; g++) {
        for (size_t r = 0; r < m_width; r++) {
          element.get()[i] = b;
          element.get()[i + 1] = g;
          element.get()[i + 2] = r;
          i += 3;
        }
      }
      m_data.push_back(element);
    }
  }

  ~DataSet() {}
};

}  // singleton

// utility function that creates 2D Node based on singleton data
auto getBuffer2D(size_t i) -> decltype(visioncpp::terminal<
    visioncpp::pixel::U8C3, common::singleton::DataSet::m_width,
    common::singleton::DataSet::m_height, visioncpp::memory_type::Buffer2D>(
    common::singleton::DataSet::Instance().m_data[i].get())) {
  return visioncpp::terminal<
      visioncpp::pixel::U8C3, common::singleton::DataSet::m_width,
      common::singleton::DataSet::m_height, visioncpp::memory_type::Buffer2D>(
      common::singleton::DataSet::Instance().m_data[i].get());
}

}  // internal

#endif  // VISIONCPP_TESTS_INCLUDE_COMMON_HPP_

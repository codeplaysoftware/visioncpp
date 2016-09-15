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

#include "../../include/common.hpp"

template <size_t TERMINAL, size_t POLICY, typename QUEUE, typename DATA>
void run_test(QUEUE &q, DATA data, int i) {
  // custom filter
  float filter_array[9] = {1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                           1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                           1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0};
  cv::Mat ref;
  // 1) load in data
  cv::Mat frame(common::singleton::DataSet::Instance().m_height,
                common::singleton::DataSet::Instance().m_width, CV_8UC3,
                common::singleton::DataSet::Instance().m_data[i].get());

  std::shared_ptr<unsigned char> ret_val(
      new unsigned char[common::singleton::DataSet::Instance().m_height *
                        common::singleton::DataSet::Instance().m_width * 3],
      [](unsigned char *dataMem) { delete[] dataMem; });
  // 2) create gold_standard image
  int kernel_size = 3;
  cv::Mat intermid;
  cv::Mat kernel = cv::Mat::ones(kernel_size, kernel_size, CV_32F) /
                   (float)(kernel_size * kernel_size);
  filter2D(intermid, ref, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);

  {
    // 3) define graph
    auto return_node = visioncpp::terminal<
        visioncpp::pixel::U8C3, common::singleton::DataSet::m_height,
        common::singleton::DataSet::m_width, visioncpp::memory_type::Buffer2D>(
        ret_val.get());

    auto node = visioncpp::point_operation<visioncpp::OP_CVBGRToRGB>(data);
    auto filter_node =
        visioncpp::terminal<float, 3, 3, visioncpp::memory_type::Buffer2D,
                            visioncpp::scope::Constant>(filter_array);
    auto node2 = visioncpp::neighbour_operation<visioncpp::OP_Filter2D>(
        node, filter_node);
    auto node3 = visioncpp::point_operation<visioncpp::OP_RGBToCVBGR>(node2);

    // assign data from node to return_node
    auto assign_node = visioncpp::assign(return_node, node3);
    // 4) execute pipe
    visioncpp::execute<POLICY, 16, 16, 8, 8>(assign_node, q);
  }
  // 7) verify
  verify(ref, ret_val);
}

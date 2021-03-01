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
#include "opencv2/core.hpp"

template <size_t TERMINAL, size_t POLICY, typename QUEUE, typename DATA>
void run_test(QUEUE &q, DATA data, int i) {
  cv::Mat ref;
  // 1) load in data
  cv::Mat frame(common::singleton::DataSet::Instance().m_height,
                common::singleton::DataSet::Instance().m_width, CV_8UC3,
                common::singleton::DataSet::Instance().m_data[i].get());

  std::shared_ptr<unsigned char> ret_val(
      new unsigned char[common::singleton::DataSet::Instance().m_height *
                        common::singleton::DataSet::Instance().m_width],
      [](unsigned char *dataMem) { delete[] dataMem; });

  // 2) create gold_standard image
  cvtColor(frame, ref, cv::COLOR_BGR2GRAY);
  {
    // 3) define graph
    auto node = visioncpp::point_operation<visioncpp::OP_CVBGRToRGB>(data);
    auto return_node = visioncpp::terminal<
        visioncpp::pixel::U8C1, common::singleton::DataSet::m_width,
        common::singleton::DataSet::m_height, visioncpp::memory_type::Buffer2D>(
        ret_val.get());

    auto node2 = visioncpp::point_operation<visioncpp::OP_RGBToGREY>(node);
    auto node3 = visioncpp::point_operation<visioncpp::OP_GREYToCVBGR>(node2);

    auto assign_node = visioncpp::assign(return_node, node3);
    // 4) execute pipe
    visioncpp::execute<POLICY, 16, 16, 8, 8>(assign_node, q);
  }
  // 7) verify
  verify(ref, ret_val);
}

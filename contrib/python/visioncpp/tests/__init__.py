from __future__ import print_function

from unittest import TestCase

import visioncpp as vp
from visioncpp import codegen


class test_codegen(TestCase):
    def init_bad_path(self):
        with self.assertRaises(vp.VisionCppException):
            vp.init("/not/a/real/path/i/think")


    def test_bad_device(self):
        node_in = vp.Image("examples/lena.jpg")
        node_out = vp.show(node_in)
        with self.assertRaises(vp.VisionCppException):
            codegen.generate(node_out, "bad_device", use_clang_format=False)


    def test_load_and_show_cpu(self):
        node_in = vp.Image("examples/lena.jpg")
        node_out = vp.show(node_in)
        self.assertEqual(codegen.generate(node_out, "cpu",
                                          use_clang_format=False),
"""#include <opencv2/opencv.hpp>
#include <visioncpp.hpp>

extern "C" {

int native_expression_tree() {
auto device = visioncpp::make_device<visioncpp::backend::sycl, visioncpp::device::cpu>();

// inputs:
std::shared_ptr<unsigned char> Image_1_data(new unsigned char[786432], [](unsigned char *d) { delete[] d; });
cv::Mat show_1_cv(512, 512, CV_8UC(3), Image_1_data.get());

{  // compute scope
cv::Mat Image_1_cv = cv::imread("examples/lena.jpg");
if (!Image_1_cv.data) {
std::cerr << "Could not open or find the image examples/lena.jpg" << std::endl;
return 1;
}
auto Image_1 = visioncpp::terminal<visioncpp::pixel::U8C3, 512, 512, visioncpp::memory_type::Buffer2D>(Image_1_cv.data);
auto Image_1_out = visioncpp::terminal<visioncpp::pixel::U8C3, 512, 512, visioncpp::memory_type::Buffer2D>(Image_1_data.get());
auto show_1 = visioncpp::assign(Image_1_out, Image_1);
visioncpp::execute<visioncpp::policy::Fuse, 16, 16, 8, 8>(show_1, device);
}  // compute scope

// outputs:
cv::namedWindow("show_1", cv::WINDOW_AUTOSIZE);
cv::imshow("show_1", show_1_cv);
cv::waitKey(0);
}

}  // extern "C"
""")


    def test_load_and_show_gpu(self):
        node_in = vp.Image("examples/lena.jpg")
        node_out = vp.show(node_in)
        self.assertEqual(codegen.generate(node_out, "gpu",
                                          use_clang_format=False),
"""#include <opencv2/opencv.hpp>
#include <visioncpp.hpp>

extern "C" {

int native_expression_tree() {
auto device = visioncpp::make_device<visioncpp::backend::sycl, visioncpp::device::gpu>();

// inputs:
std::shared_ptr<unsigned char> Image_1_data(new unsigned char[786432], [](unsigned char *d) { delete[] d; });
cv::Mat show_1_cv(512, 512, CV_8UC(3), Image_1_data.get());

{  // compute scope
cv::Mat Image_1_cv = cv::imread("examples/lena.jpg");
if (!Image_1_cv.data) {
std::cerr << "Could not open or find the image examples/lena.jpg" << std::endl;
return 1;
}
auto Image_1 = visioncpp::terminal<visioncpp::pixel::U8C3, 512, 512, visioncpp::memory_type::Buffer2D>(Image_1_cv.data);
auto Image_1_out = visioncpp::terminal<visioncpp::pixel::U8C3, 512, 512, visioncpp::memory_type::Buffer2D>(Image_1_data.get());
auto show_1 = visioncpp::assign(Image_1_out, Image_1);
visioncpp::execute<visioncpp::policy::Fuse, 16, 16, 8, 8>(show_1, device);
}  // compute scope

// outputs:
cv::namedWindow("show_1", cv::WINDOW_AUTOSIZE);
cv::imshow("show_1", show_1_cv);
cv::waitKey(0);
}

}  // extern "C"
""")

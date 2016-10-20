from __future__ import print_function

from unittest import TestCase

import visioncpp as vp
from visioncpp import codegen

class test_codegen(TestCase):
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

    def test_local_and_show_webcam_gpu(self):
        node_in = vp.Webcam()
        node_out = vp.show(node_in)
        self.assertEqual(
                codegen.generate(node_out, "gpu", use_clang_format=False),
                """#include <opencv2/opencv.hpp>
#include <visioncpp.hpp>

extern "C" {

int native_expression_tree() {
auto device = visioncpp::make_device<visioncpp::backend::sycl, visioncpp::device::gpu>();

// inputs:
std::shared_ptr<unsigned char> Webcam_1_data(new unsigned char[921600], [](unsigned char *d) { delete[] d; });
cv::VideoCapture Webcam_1_cap(0);
if (!Webcam_1_cap.isOpened()) {
  std::cerr << "Could not open capture device 0" << std::endl;
return 1;
}
cv::Mat Webcam_1_cv;
cv::Mat show_1_cv(480, 640, CV_8UC(3), Webcam_1_data.get());
for (;;) {  // main loop

{  // compute scope
Webcam_1_cap.read(Webcam_1_cv);
auto Webcam_1 = visioncpp::terminal<visioncpp::pixel::U8C3, 640, 480, visioncpp::memory_type::Buffer2D>(Webcam_1_cv.data);
auto Webcam_1_out = visioncpp::terminal<visioncpp::pixel::U8C3, 640, 480, visioncpp::memory_type::Buffer2D>(Webcam_1_data.get());
auto show_1 = visioncpp::assign(Webcam_1_out, Webcam_1);
visioncpp::execute<visioncpp::policy::Fuse, 16, 16, 8, 8>(show_1, device);
}  // compute scope

// outputs:
cv::namedWindow("show_1", cv::WINDOW_AUTOSIZE);
cv::imshow("show_1", show_1_cv);
if (cv::waitKey(1) >= 0) break;
}  // main loop
}

}  // extern "C"
""")

    def test_hello_world_image_cpu(self):
        image_in = vp.Webcam()
        node1 = vp.BGR_to_RGB(image_in)
        node2 = vp.U8C3_to_F32C3(node1)
        node3 = vp.RGB_to_HSV(node2)
        node4 = vp.HSV_to_RGB(node3)
        node5 = vp.F32C3_to_U8C3(node4)
        node6 = vp.RGB_to_BGR(node5)
        node_out = vp.show(node6)
        self.assertEqual(
            codegen.generate(node_out, "cpu", use_clang_format=False),
            """#include <opencv2/opencv.hpp>
#include <visioncpp.hpp>

extern "C" {

int native_expression_tree() {
auto device = visioncpp::make_device<visioncpp::backend::sycl, visioncpp::device::cpu>();

// inputs:
std::shared_ptr<unsigned char> Webcam_1_data(new unsigned char[921600], [](unsigned char *d) { delete[] d; });
cv::VideoCapture Webcam_1_cap(0);
if (!Webcam_1_cap.isOpened()) {
  std::cerr << "Could not open capture device 0" << std::endl;
return 1;
}
cv::Mat Webcam_1_cv;
cv::Mat show_1_cv(480, 640, CV_8UC(3), Webcam_1_data.get());
for (;;) {  // main loop

{  // compute scope
Webcam_1_cap.read(Webcam_1_cv);
auto Webcam_1 = visioncpp::terminal<visioncpp::pixel::U8C3, 640, 480, visioncpp::memory_type::Buffer2D>(Webcam_1_cv.data);
auto Webcam_1_out = visioncpp::terminal<visioncpp::pixel::U8C3, 640, 480, visioncpp::memory_type::Buffer2D>(Webcam_1_data.get());
auto BGR_to_RGB_1 = visioncpp::point_operation<visioncpp::OP_BGRToRGB>(Webcam_1);
auto U8C3_to_F32C3_1 = visioncpp::point_operation<visioncpp::OP_U8C3ToF32C3>(BGR_to_RGB_1);
auto RGB_to_HSV_1 = visioncpp::point_operation<visioncpp::OP_RGBToHSV>(U8C3_to_F32C3_1);
auto HSV_to_RGB_1 = visioncpp::point_operation<visioncpp::OP_HSVToRGB>(RGB_to_HSV_1);
auto F32C3_to_U8C3_1 = visioncpp::point_operation<visioncpp::OP_F32C3ToU8C3>(HSV_to_RGB_1);
auto RGB_to_BGR_1 = visioncpp::point_operation<visioncpp::OP_RGBToBGR>(F32C3_to_U8C3_1);
auto show_1 = visioncpp::assign(Webcam_1_out, RGB_to_BGR_1);
visioncpp::execute<visioncpp::policy::Fuse, 16, 16, 8, 8>(show_1, device);
}  // compute scope

// outputs:
cv::namedWindow("show_1", cv::WINDOW_AUTOSIZE);
cv::imshow("show_1", show_1_cv);
if (cv::waitKey(1) >= 0) break;
}  // main loop
}

}  // extern "C"
""")

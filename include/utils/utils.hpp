
#include <cstdio>
#include <memory>
#include <string>

#ifdef USE_CIMG
#include "CImg.h"
using namespace cimg_library;
#else
#include <opencv2/opencv.hpp>
#endif

namespace visioncpp {
namespace utils {

template <int COLS, int ROWS, int CHANNELS, typename T>
class IOHandler {
 public:
  IOHandler(char* filename) {
    // creating a pointer to store the results
    output_ptr =
        std::shared_ptr<T>(new unsigned char[COLS * ROWS * CHANNELS],
                           [](unsigned char* dataMem) { delete[] dataMem; });

#ifdef USE_CIMG

    printf("Using CImg\n");

    CImg<unsigned char> input(filename);
    input.resize(COLS, ROWS);
    input = input.RGBtoYCbCr().channel(0);
    input_ptr = input.data();
    // output_ptr = output.get();

    outputImage =
        CImg<unsigned char>(output_ptr.get(), COLS, ROWS, 1, CHANNELS, true);
#else

    printf("Using OpenCV\n");

    cv::Mat input = cv::imread(filename, -1);
    cv::resize(input, input, cv::Size(COLS, ROWS));
    input_ptr = input.data;

    int CV_FLAG;
    switch (CHANNELS) {
      case 1:
        CV_FLAG = CV_8UC1;
      case 2:
        CV_FLAG = CV_8UC2;
      case 3:
        CV_FLAG = CV_8UC3;
      default:
        CV_FLAG = CV_8UC3;
    }
    outputImage = cv::Mat(ROWS, COLS, CV_FLAG, output_ptr.get());

#endif
  }

  void save(char* output_file) {
#ifdef USE_CIMG
    outputImage.save(output_file);
#else
    cv::imwrite(output_file, outputImage);
    cv::imshow("Edge", outputImage);
    cv::waitKey(0);
#endif
  }

  T* getInputPointer() { return input_ptr; }

  T* getOutputPointer() { return output_ptr.get(); }

 private:
  std::shared_ptr<T> output_ptr;
  T* input_ptr;
#ifdef USE_CIMG
  CImg<unsigned char> outputImage;
#else
  cv::Mat outputImage;
#endif
};
}  // namespace utils
}  // namespace visioncpp

<div align="center">
  <br /><img src="https://www.codeplay.com/public/uploaded/public/computevision.png"><br />
</div>
# Overview
**VisionCpp** is a lightweight header-only library for computer vision and image processing.
The aim of the library is to provide a toolbox that enables performance portability for heterogeneous platforms using modern C++.

Written using [SYCL 1.2](https://www.khronos.org/registry/sycl/specs/sycl-1.2.pdf) and compiled/tested with [ComputeCpp](https://codeplay.com/products/computesuite/computecpp) to accelerate vision code using OpenCL devices.

## Table of contents
* [Integration](#integration)
* [Sample Code](#sample-code)
* [Requirements](#requirements)
* [Build](#build)
* [Examples](#examples)
* [Documentation](#documentation)
* [Contributing](#contributing)
* [Resources](#resources)
* [License](#license)
* [Known Issues](#known-issues)

## <a name="integration" /> Integration
All you need to do is include the VisionCpp.hpp header in your project and you are good to go! ( assuming that OpenCL and ComputeCPP is installed correctly. )

~~~~~~~~~~~~~~~{.cpp}
#include <visioncpp.hpp> //all that is needed
~~~~~~~~~~~~~~~

## <a name="sample-code" /> Sample Code
Below is a very simple application that will do the conversion RGB -> HSV. Full source code can be found in the examples folder.
RGB is assumed to be a three-channel unsigned char storage with a reasonable channel order.

~~~~~~~~~~~~~~~{.cpp}
  // main, args, checks and all the boring stuff

  // ...

  // where VisionCpp will run.
  auto dev = visioncpp::make_device<visioncpp::backend::sycl,
                                    visioncpp::device::cpu>();

  // create a host container for input data
  std::shared_ptr<unsigned char> in_rgb(new unsigned char[3],
  [](unsigned char *dataMem) { delete[] dataMem;});

  in_rgb.get()[0] = atoi(argv[1]);
  in_rgb.get()[1] = atoi(argv[2]);
  in_rgb.get()[2] = atoi(argv[3]);

  // create a host container for output data
  std::shared_ptr<unsigned char> out_hsv(new unsigned char[3],
  [](unsigned char *dataMem) { delete[] dataMem;});

  // exiting this scope will sync data
  {
    // definition of the VisionCpp pipeline:

    // create terminal nodes - a leaf node ( data node ) of the expression tree.
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
        visioncpp::terminal<visioncpp::pixel::U8C3, 1, 1,
                            visioncpp::memory_type::Buffer2D>(out_hsv.get());

    // unsigned char -> float RGB storage conversion
    auto node = visioncpp::point_operation<visioncpp::OP_U8C3ToF32C3>(data);
    // float RGB to float HSV conversion
    auto node2 = visioncpp::point_operation<visioncpp::OP_RGBToHSV>(node);
    // helper node that allows display of HSV
    // for unsigned char: V <- 255*V, S <- 255*S, H <- H/2 ( to fit in range of 0..255 )
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

~~~~~~~~~~~~~~~

## <a name="requirements" /> Requirements
To successfully compile VisionCpp tests, you will need:
* ComputeCpp (https://codeplay.com/products/computesuite/computecpp)
* OpenCV 3.2 (https://github.com/opencv/opencv) - used for camera access, window display and as a testing reference.
* GTest (https://github.com/google/googletest) - testing framework.
* OpenCL 1.2

## <a name="build" /> Build
Assuming you are in the root of a git repo:
~~~~~~~~~~~~~~~{.sh}
mkdir build
cd  build
cmake .. -DCOMPUTECPP_PACKAGE_ROOT_DIR={PATH_TO_COMPUTECPP_ROOT} -DCMAKE_CXX_COMPILER={FAVORITE_CXX_COMPILER}
make -j8
make test
~~~~~~~~~~~~~~~

The output binaries will be catalogued in bin folder.
~~~~~~~~~~~~~~~{.sh}
| - build
  | - bin
    | - example
    | - test
~~~~~~~~~~~~~~~

## <a name="examples" /> Examples
Most of the examples are using camera.

## <a name="documentation" /> Documentation
Online documentation can be found [here](https://codeplaysoftware.github.io/visioncpp/).

The documentation is created using [Doxygen](http://www.stack.nl/~dimitri/doxygen/).
~~~~~~~~~~~~~~~{.sh}
make doc
~~~~~~~~~~~~~~~

The documentation will be created in html folder in build directory.
~~~~~~~~~~~~~~~{.sh}
| - build
  | - doc
~~~~~~~~~~~~~~~

## <a name="contributing" /> Contributing
Contributors always welcome! See <a href="https://github.com/codeplaysoftware/visioncpp/blob/master/CONTRIBUTING.md">CONTRIBUTING.md</a> for details.

The list of <a href="https://github.com/codeplaysoftware/visioncpp/blob/master/CONTRIBUTORS.md">contributors</a>.

## <a name="resources" /> Resources
* [Kernel composition in SYCL](http://opus.bath.ac.uk/49695/)
* [VisionCPP: A SYCL-based Computer Vision Framework](http://dl.acm.org/citation.cfm?doid=2909437.2909444)
* [Wiki](https://github.com/codeplaysoftware/visioncpp/wiki)

## <a name="license" /> License
The Apache License, Version 2.0 License. See [LICENSE](https://github.com/codeplaysoftware/visioncpp/blob/master/LICENSE) for more.

## <a name="known-issues" /> Known Issues
* The Tuple class works only with clang++.

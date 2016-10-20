#!/usr/bin/env python

"""
Python interface to VisionCpp.
"""
__author__ = "Chris Cummins"
__email__ = "chrisc.101@gmail.com"
__copyright__ = "Copyright 2016 Codeplay Software Limited"
__license__ = "Apache License, Version 2.0"

import os
import logging


class VisionCppException(Exception):
    """
    VisionCpp module exception.
    """
    pass


from visioncpp import backend
from visioncpp import codegen
from visioncpp import util


default_computecpp_prefix = "/usr/local"
computecpp_prefix = default_computecpp_prefix


def init(path=None):
    """
    Initialize VisionCpp module.

    Arguments:
        path (str, optional): Path to ComputeCpp directory.

    Returns:
        str: Path to ComputeCpp directory.
    """
    def must_exist(path):
        if not os.path.exists(path):
            raise VisionCppException(
                "File {} not found. Is ComputeCpp installed?".format(path))

    if path is not None:
        global computecpp_prefix
        path = os.path.abspath(os.path.expanduser(path))
        must_exist(path)
        computecpp_prefix = path

    return computecpp_prefix


def run(expression, devtype="cpu"):
    """
    Execute a VisionCpp expression tree.

    Arguments:
        expression (Operation): The final node of the expression tree.
        devtype (string, optional): Execution device.

    Returns:
        object: Expression output.
    """
    code = codegen.generate(expression, devtype)
    binary = backend.compile_cpp_code(code)
    output = backend.run_binary(binary)
    return output


# Operations Types:

class Operation(object):
    """
    VisionCpp base expression type.
    """
    def __init__(self):
        self.parent = None
        self.name = None

    def _input_code(self):
        """ Get code lines for inputs """
        return None

    def _compute_code(self):
        """ Get code lines for compute scope """
        return None

    def _output_code(self):
        """ Get code lines for outputs """
        return None

    def __repr__(self):
        return '{node}<{base}>'.format(node=type(self).__name__,
                                       base=repr(self.parent))

    def __str__(self):
        return str(self.__repr__())


class TerminalOperation(Operation):
    """
    VisionCpp terminal operation type.
    """
    parent = None


class PointOperation(Operation):
    """
    VisionCpp pointwise operation type.
    """
    def _compute_code(self):
        return [
            "auto {name} = visioncpp::point_operation<"
            "visioncpp::OP_{type}>({parent});"
            .format(name=self.name,
                    type=type(self).__name__,
                    parent=self.parent.name)
        ]


class NeighbourOperation(Operation):
    """
    VisionCpp neighbour operation type.
    """
    pass


# Terminal Operations:

def _shared_ptr(name, nbytes):
    """ Generate a shared pointer. """
    return [
        ("std::shared_ptr<unsigned char> {name}("
         "new unsigned char[{nbytes}], "
         "[](unsigned char *d) {{ delete[] d; }});"
         .format(name=name, nbytes=nbytes))
    ]


class Image(TerminalOperation):
    """
    An image node.
    """
    def __init__(self, path):
        if not path.endswith(".jpg"):
            raise VisionCppException("Unspported image type")

        self.input = os.path.expanduser(path)
        if not os.path.exists(self.input):
            raise VisionCppException(
                "Image file '{}' not found".format(self.input))

        # TODO: Use opencv to get image properties.
        self.width, self.height = util.get_image_size(self.input)
        self.channels = 3

    def _input_code(self):
        nbytes = self.width * self.height * self.channels
        return _shared_ptr(self.name + "_data", nbytes)

    def _compute_code(self):
        return [
            "cv::Mat {name}_cv = cv::imread(\"{arg}\");".format(
                name=self.name, arg=self.input),
            "if (!{name}_cv.data) {{".format(name=self.name),
            ("std::cerr << \"Could not open or find the image {arg}\" "
             "<< std::endl;".format(arg=self.input)),
            "return 1;",
            "}",
            ("auto {name} = visioncpp::terminal<visioncpp::pixel::U8C3, "
             "{width}, {height}, visioncpp::memory_type::Buffer2D>"
             "({name}_cv.data);").format(
                name=self.name, width=self.width, height=self.height),
            ("auto {name}_out = visioncpp::terminal<visioncpp::pixel::U8C3, "
             "{width}, {height}, visioncpp::memory_type::Buffer2D>"
             "({name}_data.get());").format(
                name=self.name, width=self.width, height=self.height)
        ]

    def __repr__(self):
        return "Image<{}>".format(self.input)


class Webcam(TerminalOperation):
    """
    A webcam input feed.
    """
    def __init__(self, device_id=0):
        """
        Construct a Webcam node.

        Arguments:
            device_id (int, optional): Capture device ID.
        """
        self.device_id = device_id

        # TODO: Use opencv to get capture properties.
        self.width = 640
        self.height = 480
        self.channels = 3

    def _input_code(self):
        nbytes = self.width * self.height * self.channels
        return _shared_ptr(self.name + "_data", nbytes) + [
            "cv::VideoCapture {name}_cap({devid});".format(name=self.name,
                                                           devid=self.device_id),
            "if (!{name}_cap.isOpened()) {{".format(name=self.name),
            ("  std::cerr << \"Could not open capture device {devid}\" "
             "<< std::endl;").format(devid=self.device_id),
            "return 1;",
            "}",
            "cv::Mat {name}_cv;".format(name=self.name),
        ]

    def _compute_code(self):
        return [
            "{name}_cap.read({name}_cv);".format(name=self.name),
            ("auto {name} = visioncpp::terminal<visioncpp::pixel::U8C3, "
             "{width}, {height}, visioncpp::memory_type::Buffer2D>"
             "({name}_cv.data);").format(
                name=self.name, width=self.width, height=self.height),
            ("auto {name}_out = visioncpp::terminal<visioncpp::pixel::U8C3, "
             "{width}, {height}, visioncpp::memory_type::Buffer2D>"
             "({name}_data.get());").format(
                name=self.name, width=self.width, height=self.height)
        ]

    def __repr__(self):
        return "Webcam<{}>".format(self.device_id)


class show(TerminalOperation):
    """
    Node to display an image.
    """
    def __init__(self, parent):
        self.parent = parent
        self.repeating = False

        input = self.parent
        while input:
            if isinstance(input, Image):
                break
            elif isinstance(input, Webcam):
                self.repeating = True
                break
            input = input.parent

        if not input:
            raise VisionCppException("Expression tree has no input")
        self.input = input

    def _input_code(self):
        return [
            ('cv::Mat {name}_cv({height}, {width}, CV_8UC({channels}), '
             '{input}_data.get());'.format(
                name=self.name, width=self.input.width,
                height=self.input.height,
                channels=self.input.channels, input=self.input.name)),
        ]

    def _compute_code(self):
        return [
            ("auto {name} = visioncpp::assign({input}_out, {tail});"
             .format(name=self.name, input=self.input.name,
                     tail=self.parent.name)),
            ("visioncpp::execute<visioncpp::policy::Fuse, 16, 16, 8, 8>("
             "{name}, device);".format(name=self.name))
        ]

    def _output_code(self):
        lines = [
            ('cv::namedWindow("{name}", cv::WINDOW_AUTOSIZE);'
             .format(name=self.name)),
            'cv::imshow("{name}", {name}_cv);'.format(name=self.name),
        ]

        if self.repeating:
            lines += ["if (cv::waitKey(1) >= 0) break;"]
        else:
            lines += ["cv::waitKey(0);"]

        return lines


# PointOperations:

class BGRToRGB(PointOperation):
    """
    This functor reorders channels BGR to RGB.
    """
    def __init__(self, parent):
        self.parent = parent


class RGBToHSV(PointOperation):
    """
    Functor converts RGB to HSV color space.
    """
    def __init__(self, parent):
        self.parent = parent


class RGBToBGR(PointOperation):
    """
    This functor reorders channels BGR to RGB.
    """
    def __init__(self, parent):
        self.parent = parent


class HSVToRGB(PointOperation):
    """
    Functor converts HSV to color RGB.
    """
    def __init__(self, parent):
        self.parent = parent


class U8C3ToF32C3(PointOperation):
    """
    This functor performs conversion from [0, 255] to [0.0f, 1.0f].
    """
    def __init__(self, parent):
        self.parent = parent


class F32C3ToU8C3(PointOperation):
    """
    This functor performs conversion from [0.0f, 1.0f] to [0, 255].
    """
    def __init__(self, parent):
        self.parent = parent

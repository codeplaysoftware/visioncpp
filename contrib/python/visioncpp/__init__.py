#!/usr/bin/env python

"""
Python interface to VisionCpp.
"""
import os

__author__ = "Chris Cummins"
__email__ = "chrisc.101@gmail.com"
__copyright__ = "Copyright 2016 Codeplay Software Limited"
__license__ = "Apache License, Version 2.0"

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
    def __init__(self, parent):
        self.parent = parent

    def _compute_code(self):
        return [
            "auto {name} = visioncpp::point_operation<"
            "visioncpp::OP_{type}>({parent});"
            .format(name=self.name,
                    type=type(self).__name__,
                    parent=self.parent.name)
        ]


class OpWithArgs(Operation):
    """
    VisionCpp pointwise operation type.
    """
    def __init__(self, *args):
        assert(len(args) >= 1)
        self.parent = args[0]
        self.args = args

    def _compute_code(self):
        return [
            "auto {name} = visioncpp::point_operation<"
            "visioncpp::OP_{type}>({argnames});"
            .format(name=self.name,
                    type=type(self).__name__,
                    argnames=", ".join([x.name for x in self.args]))
        ]


class NeighbourOperation(Operation):
    """
    VisionCpp neighbour operation type.
    """
    def __init__(self, parent):
        self.parent = parent

    def _compute_code(self):
        return [
            "auto {name} = visioncpp::point_operation<"
            "visioncpp::OP_{type}>({parent});"
            .format(name=self.name,
                    type=type(self).__name__,
                    parent=self.parent.name)
        ]


class NeighbourOpWithArg(Operation):
    """
    VisionCpp neighbour operation type.
    """
    def __init__(self, parent, arg):
        self.parent = parent
        self.arg = arg

    def _compute_code(self):
        return [
            "auto {name} = visioncpp::point_operation<"
            "visioncpp::OP_{type}>({parent}, {arg});"
            .format(name=self.name,
                    type=type(self).__name__,
                    parent=self.parent.name,
                    arg=self.arg.name)
        ]

    def __repr__(self):
        return '{node}<{base}, {arg}>'.format(
            node=type(self).__name__,
            base=repr(self.parent),
            arg=repr(self.arg))


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
            "cv::VideoCapture {name}_cap({devid});".format(
                name=self.name, devid=self.device_id),
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
    pass


class F32C3ToU8C3(PointOperation):
    """
    This functor performs conversion from [0.0f, 1.0f] to [0, 255].
    """
    pass


class HSVToRGB(PointOperation):
    """
    Functor converts HSV to color RGB.
    """
    pass


class HSVToU8C3(PointOperation):
    """
    Functor allows displaying HSV.
    """
    pass


class RGBToBGR(PointOperation):
    """
    This functor reorders channels BGR to RGB.
    """
    pass


class RGBToGREY(PointOperation):
    """
    This functor performs RGB to GREY convertion.

    Uses the following rule: GREY <- 0.299f * R + 0,587f * G + 0.114 * B.
    """
    pass


class RGBToHSV(PointOperation):
    """
    Functor converts RGB to HSV color space.
    """
    pass


class U8C3ToF32C3(PointOperation):
    """
    This functor performs conversion from [0, 255] to [0.0f, 1.0f].
    """
    pass


class Filter2D(NeighbourOpWithArg):
    """
    Applying the general convolution for 3 channel Image.
    """
    pass


class Filter2D_One(NeighbourOpWithArg):
    """
    Applying the general convolution for 1 channel Image.
    """
    pass


class GaussianBlur3x3(NeighbourOperation):
    """
    Applying the Gaussian blur 3x3.
    """
    pass


class SepFilterRow(NeighbourOpWithArg):
    """
    Separable filter for rows.
    """
    pass


class SepFilterCol(NeighbourOpWithArg):
    """
    Separable filter for cols.
    """
    pass


class SepGaussRow3(NeighbourOperation):
    """
    Applying the general convolution for 3 channel Image.
    """
    pass


class SepGaussCol3(NeighbourOperation):
    """
    Applying the general convolution for 3 channel Image.
    """
    pass


class DownsampleAverage(NeighbourOperation):
    """
    Downsampling filter using average technique Other filters could be added
    for different numbers of channels.
    """
    pass


class DownsampleClosest(NeighbourOperation):
    """
    Downsampling filter using closest technique.
    """
    pass


class AbsSub(OpWithArgs):
    """
    Uses the sycl::fabs to return the difference.
    """
    pass


class Add(OpWithArgs):
    """
    This functor adds two pixels.
    """
    pass


class AniDiff_Grey(OpWithArgs):
    """
    This functor applies anisotropic diffusion for one channel.
    """
    pass


class AniDiff(OpWithArgs):
    """
    This functor applies anisotropic diffusion for 3 channels.
    """
    pass


class Div(OpWithArgs):
    """
    This functor divides two matrices element-wise.
    """
    pass


class FloatToF32C3(OpWithArgs):
    """
    It replicates one channel to 3 channels.
    """
    pass


class FloatToU8C1(OpWithArgs):
    """
    It converts float to uchar converting [0.0f, 1.0f] to [0, 255].
    """
    pass


class U8C1ToFloat(OpWithArgs):
    """
    It converts uchar to float converting range [0, 255] to [0.0f, 1.0f].
    """
    pass


class FloatToUChar(OpWithArgs):
    """
    It converts float to uchar.
    """
    pass


class Median(OpWithArgs):
    """
    This functor implements a median filter.
    """
    pass


class Merge2Chns(OpWithArgs):
    """
    This functor merges 2 matrices into one matrix of 2 channels.
    """
    pass


class Mul(OpWithArgs):
    """
    This functor implements an element-wise matrix multiplication.
    """
    pass


class PowerOf2(OpWithArgs):
    """
    This functor implements the power of 2 of one matrix.
    """
    pass


class Scale(OpWithArgs):
    """
    Scales the pixel value of an image by a factor.
    """
    pass


class Sub(OpWithArgs):
    """
    This functor subtracts 2 images.
    """
    pass


class Thresh(OpWithArgs):
    """
    Implements a binary threshold.
    """
    pass


class Broadcast(OpWithArgs):
    """
    This functor sets the pixel to the value passed in.
    """
    pass


class ScaleChannelZero(OpWithArgs):
    """
    This functor applies anisotropic diffusion for 3 channels.
    """
    pass


class ScaleChannelOne(OpWithArgs):
    """
    This functor applies anisotropic diffusion for 3 channels.
    """
    pass


class ScaleChannelTwo(OpWithArgs):
    """
    This custom functor changes 2 channel by factor passed via float f.
    """
    pass

from __future__ import print_function

from unittest import TestCase

import os
import visioncpp as vp


class TestData404(Exception): pass


def data_path(path, exists=True):
    """
    Return absolute path to unittest data file. Data files are located in
    tests/data.

    Args:
        path (str): Relative path.

    Returns:
        string: Absolute path.

    Raises:
        TestData404: If path doesn"t exist.
    """
    abspath = os.path.join(os.path.dirname(__file__), "data", path)
    if exists and not os.path.exists(abspath):
        raise TestData404(abspath)
    return abspath


def data_str(path):
    """
    Return contents of unittest data file as a string.

    Args:
        path (str): Relative path.

    Returns:
        string: File contents.

    Raises:
        TestData404: If path doesn't exist.
    """
    with open(data_path(path)) as infile:
        return infile.read()


_in = vp.Image(data_path('lena.jpg'))


class test_visioncpp(TestCase):
    def test_init_path_unchanged(self):
        path = vp.init()
        path2 = vp.init()
        self.assertEqual(path, path2)

    def test_init_path_set(self):
        path = vp.init('README.md')
        self.assertEqual(path, os.path.abspath('README.md'))
        path2 = vp.init()
        self.assertEqual(path, path2)

    def test_init_bad_path(self):
        with self.assertRaises(vp.VisionCppException):
            vp.init("/not/a/real/path/i/think")


class test_Operation(TestCase):
    def test_interface(self):
        op = vp.Operation()
        self.assertEqual(None, op._input_code())
        self.assertEqual(None, op._compute_code())
        self.assertEqual(None, op._output_code())


class test_PointOperation(TestCase):
    OPS = [
        vp.BGRToRGB(_in),
        vp.F32C3ToU8C3(_in),
        vp.HSVToRGB(_in),
        vp.HSVToU8C3(_in),
        vp.RGBToBGR(_in),
        vp.RGBToGREY(_in),
        vp.RGBToHSV(_in),
        vp.U8C3ToF32C3(_in),
    ]

    def test_repr(self):
        base = repr(_in)
        for node in self.OPS:
            self.assertEqual(repr(node),
                "{node}<{base}>".format(node=type(node).__name__, base=base))

    def test_compute_code(self):
        _in.name = "foo"
        for node in self.OPS:
            node.name = "bar"
            self.assertEqual(
                node._compute_code(),
                ["auto bar = visioncpp::point_operation<"
                 "visioncpp::OP_{type}>(foo);".format(
                     type=type(node).__name__)])


class test_NeighbourOpWithArg(TestCase):
    OPS = [
        vp.Filter2D(_in, _in),
        vp.Filter2D_One(_in, _in),
        vp.SepFilterRow(_in, _in),
        vp.SepFilterCol(_in, _in),
    ]

    def test_repr(self):
        base = repr(_in)
        for node in self.OPS:
            self.assertEqual(repr(node),
                "{node}<{base}, {base}>".format(
                    node=type(node).__name__, base=base))

    def test_compute_code(self):
        _in.name = "foo"
        for node in self.OPS:
            node.name = "bar"
            self.assertEqual(
                node._compute_code(),
                ["auto bar = visioncpp::point_operation<"
                 "visioncpp::OP_{type}>(foo, foo);".format(
                     type=type(node).__name__)])


class test_NeighbourOperation(TestCase):
    OPS = [
        vp.GaussianBlur3x3(_in),
        vp.SepGaussRow3(_in,),
        vp.SepGaussCol3(_in,),
        vp.DownsampleAverage(_in,),
        vp.DownsampleClosest(_in,),
    ]

    def test_repr(self):
        base = repr(_in)
        for node in self.OPS:
            self.assertEqual(repr(node),
                "{node}<{base}>".format(node=type(node).__name__, base=base))

    def test_compute_code(self):
        _in.name = "foo"
        for node in self.OPS:
            node.name = "bar"
            self.assertEqual(
                node._compute_code(),
                ["auto bar = visioncpp::point_operation<"
                 "visioncpp::OP_{type}>(foo);".format(
                     type=type(node).__name__)])


class test_Image(TestCase):
    def test_unsupported_image(self):
        with self.assertRaises(vp.VisionCppException):
            vp.Image('foo.gif')

    def test_bad_path(self):
        with self.assertRaises(vp.VisionCppException):
            vp.Image('/not/a/real/file.jpg')


class test_Webcam(TestCase):
    def test_devid(self):
        w = vp.Webcam(0)
        self.assertEqual(0, w.device_id)
        w2 = vp.Webcam(1)
        self.assertEqual(1, w2.device_id)


class test_show(TestCase):
    def test_no_input(self):
        with self.assertRaises(vp.VisionCppException):
            vp.show(None)

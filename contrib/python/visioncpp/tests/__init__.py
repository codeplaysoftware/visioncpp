from __future__ import print_function

from unittest import TestCase

import os
import visioncpp as vp


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

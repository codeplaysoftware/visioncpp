from __future__ import print_function

from unittest import TestCase

import visioncpp as vp


class test_visioncpp(TestCase):
    def test_init_bad_path(self):
        with self.assertRaises(vp.VisionCppException):
            vp.init("/not/a/real/path/i/think")


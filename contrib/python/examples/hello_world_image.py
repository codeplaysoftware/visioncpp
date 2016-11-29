#!/usr/bin/env python
#
# usage: hello_world_image.py [-h] [-v] [-C PATH]
#
# A very simple VisionCpp demo. Loads an image from file, performs some colour
# space conversions, and displays the result.
#
# optional arguments:
#   -h, --help            show this help message and exit
#   -v, --verbose         verbose output printing
#   -C PATH, --computecpp PATH
#                         Path to ComputeCpp installation directory
#
from __future__ import print_function

import visioncpp as vp
import logging
import os

from argparse import ArgumentParser
from sys import argv, stderr


def main():
    # Command line options:
    parser = ArgumentParser(description="A very simple VisionCpp demo. Loads "
                            "an image from file, performs some colour space "
                            "conversions, and displays the result.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose output printing")
    parser.add_argument("-C", "--computecpp", type=str, metavar="PATH",
                        default="~/ComputeCpp-CE-0.1-Linux",
                        help="Path to ComputeCpp installation directory")
    args = parser.parse_args()
    if args.verbose: logging.basicConfig(level=logging.DEBUG)
    vp.init(args.computecpp)

    image_path = os.path.join(os.path.dirname(__file__), "lena.jpg")

    # VisionCpp expression tree:
    image_in = vp.Image(image_path)
    node1 = vp.BGRToRGB(image_in)
    node2 = vp.U8C3ToF32C3(node1)
    node3 = vp.RGBToHSV(node2)
    node4 = vp.HSVToRGB(node3)
    node5 = vp.F32C3ToU8C3(node4)
    node6 = vp.RGBToBGR(node5)
    output = vp.show(node6)

    vp.run(output)


if __name__ == "__main__":
    debug = os.environ.get("DEBUG", False)

    if debug:
        main()
    else:
        try:
            main()
        except Exception as e:
            print(type(e).__name__ + ":", e, file=stderr)

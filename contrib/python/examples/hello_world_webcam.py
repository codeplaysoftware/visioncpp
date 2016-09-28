#!/usr/bin/env python
#
# usage: hello_world_webcam.py [-h] [-v] [-C PATH]
#
# A very simple VisionCpp demo. Opens the webcam capture feed, performs some
# colour space conversions, and displays the result.
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

from argparse import ArgumentParser
from sys import argv, stderr


def main():
    # Command line options:
    parser = ArgumentParser(description="A very simple VisionCpp demo. Opens "
                            "the webcam capture feed, performs some colour "
                            "space conversions, and displays the result.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="verbose output printing")
    parser.add_argument("-C", "--computecpp", type=str, metavar="PATH",
                        default="~/ComputeCpp-CE-0.1-Linux",
                        help="Path to ComputeCpp installation directory")
    args = parser.parse_args()
    if args.verbose: logging.basicConfig(level=logging.DEBUG)
    vp.init(args.computecpp)

    # VisionCpp expression tree:
    image_in = vp.Webcam()
    node1 = vp.BGR_to_RGB(image_in)
    node2 = vp.U8C3_to_F32C3(node1)
    node3 = vp.RGB_to_HSV(node2)
    node4 = vp.HSV_to_RGB(node3)
    node5 = vp.F32C3_to_U8C3(node4)
    node6 = vp.RGB_to_BGR(node5)
    output = vp.show(node6)

    vp.run(output)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(type(e).__name__ + ":", e, file=stderr)
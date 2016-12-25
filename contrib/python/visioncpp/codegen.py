from __future__ import print_function

import visioncpp as vp

import logging
import sys

from collections import defaultdict
from distutils.spawn import find_executable
from labm8 import types
from subprocess import Popen, PIPE

from visioncpp import util


def library_source(lines, pipeline):
    """
    Return source code for a VisionCpp library.

    The generated code will compile a callable library method
    "natvie_expression_tree()".

    Arguments:
        lines (str[]): Program implementation.
        pipeline (Node[]): Program pipeline.

    Returns:
        str: Source code
    """
    def get_input_nodes(pipeline):
        inputs = []
        for node in pipeline:
            if isinstance(node, vp.Image):
                inputs.append(node)
        return inputs

    inputs = get_input_nodes(pipeline)
    input_args = ", ".join("unsigned char *const {}_arg".format(node.name)
                           for node in inputs)

    return "\n".join([
        "#include <visioncpp.hpp>",
        "",
        "extern \"C\" {",
        "",
        "int native_expression_tree({inputs}, unsigned char *const out) {{"
        .format(inputs=input_args),
    ] + lines + [
        "  return 0;",
        "}",
        "",
        "}  // extern \"C\"\n"
    ])


def get_device(devtype="cpu", name="device"):
    """
    Return the lines to construct a VisionCPP device.

    Arguments:
        devtype (str, optional): Either "gpu" or "cpu".
        name (str, optional): Name of the constructed device variable.

    Returns:
        str[]: C++ lines.

    Raises:
        VisionCppException: If device type is neither "cpu" or "gpu".
    """
    devtype = devtype.lower()
    if devtype != "cpu" and devtype != "gpu":
        raise vp.VisionCppException("Bad device type '{}'".format(devtype))

    return [
        "auto {name} = visioncpp::make_device<"
        "visioncpp::backend::sycl, "
        "visioncpp::device::{dev}>();".format(dev=devtype, name=name)
    ]


def find_clang_format():
    """
    Find clang-format binary.

    Returns:
        str: Path to clang-format if found, else None.
    """
    clang_format = {"val": None}

    def set_if_exists(executable):
        if not clang_format["val"]:
            clang_format["val"] = find_executable(executable)

    set_if_exists("clang-format")
    set_if_exists("clang-format-3.6")
    set_if_exists("clang-format-3.8")

    return clang_format["val"]


def clang_format(code, clang_format=find_clang_format()):
    """
    Run clang-format over C++ code.

    This function is safe to call, regardless of whether clang-format is
    present on the system. If clang-format is not found, the returned code
    is the same as the input.

    Arguments:
        code (str): C++ code.
        clang_format (str, optional): Path to clang-format.

    Returns:
        str: Formatted C++ code.
    """
    if clang_format is None:
        logging.debug("Could not find clang-format. Is it installed?")
        return code

    process = Popen([clang_format], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate(code.encode('utf-8'))

    if stderr:
        print(stderr.decode("utf-8"), file=sys.stderr)
    if process.returncode != 0:
        sys.exit(1)

    return stdout.decode('utf-8')


def process_expression(root):
    """
    Process an expression tree into a linear sequence of nodes.

    Arguments:
        root (VisionCpp.Operation): Expression tree.

    Returns:
        VisionCpp.Operation []: Serialized expression tree.
    """
    name_ids = defaultdict(int)
    pipeline = []
    node = root
    while node is not None:
        # Generate a unique node name:
        typename = type(node).__name__
        name_ids[typename] += 1
        node.name = "{type}_{id}".format(type=typename, id=name_ids[typename])

        pipeline.append(node)
        node = node.parent
    return list(reversed(pipeline))


def generate(expression, devtype, use_clang_format=True):
    """
    Generate C++ code for an expression tree.

    Arguments:
        expression (VisionCpp.Operation): Root of the expression tree.
        devtype (str, optional): Execution device type.

    Returns:
        tuple (list of VisionCpp.Operation, str): Pipeline and C++ code.
    """
    if not isinstance(expression, vp.Operation): raise TypeError
    if not types.is_str(devtype): raise TypeError

    pipeline = process_expression(expression)

    lines = []
    lines += get_device(devtype)

    def append_node(lines, node, attr):
        line = util.call_if_attribute(node, attr)
        if line:
            lines += line

    def get_pipeline_stage(lines, pipeline, stage):
        for node in pipeline:
            append_node(lines, node, "_{}_code".format(stage))

    # Inputs:
    lines += ["\n// inputs:"]
    get_pipeline_stage(lines, pipeline, "input")

    # Compute scope:
    lines += ["\n{  // compute scope"]
    get_pipeline_stage(lines, pipeline, "compute")
    lines += ["}  // compute scope"]

    # Outputs:
    lines += ["\n// outputs:"]
    get_pipeline_stage(lines, pipeline, "output")

    code = library_source(lines, pipeline)

    if use_clang_format:
        code = clang_format(code)

    logging.info("Generated C++ code:\n" + code)
    return pipeline, code

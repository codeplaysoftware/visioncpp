# Python interface for VisionCpp
[![Build Status](https://travis-ci.org/ChrisCummins/visioncpp.svg?branch=master)](https://travis-ci.org/ChrisCummins/visioncpp)

Provides a simple python interface for GPU-accelerated vision processing
using VisionCpp.

## Requirements

* python >= 2.7 or 3
* [ComputeCpp](https://www.codeplay.com/products/computesuite/computecpp)
* [clang-format](http://llvm.org/releases/download.html) *(optional)*

## Installation

```sh
virtualenv env
source env/bin/activate
python ./setup.py install && pip install ipython
```

## Get Started

```sh
$ ./env/bin/ipython
```

```py
>>> import visioncpp as vp
>>> vp.init("/opt/ComputeCpp-CE-0.1-Linux")  # path to your ComputeCpp package
>>> a = vp.Image("examples/lena.jpg")
>>> b = vp.show(a)
>>> vp.run(d)
```

See `examples/` directory for more information.

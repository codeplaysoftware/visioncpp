# Python interface for VisionCpp
[![Build Status](https://travis-ci.org/ChrisCummins/visioncpp.svg?branch=development)](https://travis-ci.org/ChrisCummins/visioncpp) [![Coverage Status](https://coveralls.io/repos/github/ChrisCummins/visioncpp/badge.svg?branch=development)](https://coveralls.io/github/ChrisCummins/visioncpp?branch=development)

Provides a simple python interface for GPU-accelerated vision processing
using VisionCpp.

## Requirements

* python >= 3.3, or 2.7
* [ComputeCpp](https://www.codeplay.com/products/computesuite/computecpp)
* [clang-format](http://llvm.org/releases/download.html) *(optional)*

## Installation

```
$ pip install visioncpp
```

To build from source:

```sh
$ virtualenv env
$ source env/bin/activate
(env) $ python ./setup.py install && pip install ipython
```

## Get Started

```sh
(env) $ ./env/bin/ipython
```

```py
>>> import visioncpp as vp
>>> vp.init("/opt/ComputeCpp-CE-0.1-Linux")  # path to your ComputeCpp package
>>> a = vp.Image("examples/lena.jpg")
>>> b = vp.show(a)
>>> vp.run(b)
```

See `examples/` directory for more information.
# Tutorial

This tutorial will cover the basics of setting up a VisionCpp environment and writing a simple, GPU-accelerated python program.

We begin by first installing VisionCpp. To do this, head to the [Codeplay site](https://www.codeplay.com/products/computesuite/computecpp) and download and install the *ComputeCpp Community Edition Beta*. Make a note of the installation location; for the remainder of this tutorial we will assume it has been installed in `~/computecpp`.

Next, check that you have python installed:

```sh
$ python --version
Python 3.5.2
```

For the remainder of this tutorial we will assume you are using Python 3, but other versions are supported, see [Travis CI](https://travis-ci.org/ChrisCummins/visioncpp) for a list of supported versions.

Next, we will create a python [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/) for installing VisionCpp tests:

```sh
$ cd ~
$ virtualenv visioncpp
$ source visioncpp/bin/activate
```

Your shell prefix will be prefaced with `(visioncpp)`, and any installed packages will go into this newly created virtual environment. We will now install visioncpp, and the interactive shell [ipython](https://ipython.org/):

```sh
(visioncpp) $ pip install visioncpp ipython
```

That's it! VisionCpp is now installed. Begin an ipython session to try it out:

```sh
(visioncpp) $ ./visioncpp/bin/ipython
```

We begin the python session by importing the visioncpp module:

```python
In [1]: import visioncpp as vp
```

You can call `help()` with any VisionCpp object to see the interface documentation. Try it on the main module to see an overview of the API:

```python
In [2]: help(vp)
```

Before we can use it, we need to tell the python interface where to find ComputeCpp. Do this by calling the `init()` function with the path that you made a note of earlier:

```python
In [3]: vp.init("~/computecpp")
```

VisionCpp has two main parts:

1. An interface for constructing complex vision processing jobs as expression trees.
2. A runtime for executing expression trees using optimized native code with GPU acceleration.

We'll now create an expression tree. Download this [test image](https://github.com/ChrisCummins/visioncpp/raw/development/contrib/python/examples/lena.jpg) and create a vision node for an image:

```python
In [4]: input = vp.Image("~/Downloads/lena.jpg")
```

We will then add a few nodes to the expression tree, performing some simple colour space conversions before finally showing the resulting image:

```python
In [5]: node1 = vp.BGRToRGB(input)
In [6]: node2 = vp.U8C3ToF32C3(node1)
In [7]: node3 = vp.RGBToHSV(node2)
In [8]: node4 = vp.HSVToRGB(node3)
In [9]: node5 = vp.F32C3ToU8C3(node4)
In [10]: node6 = vp.RGBToBGR(node5)
In [11]: output = vp.show(node6)
```

As you can see, expression trees are constructed from top to bottom, with each node pointing to one or more parent nodes. Expression trees are regular python objects, you can inspect them by calling print:

```python
In [12]: print(output)
show<RGBToBGR<F32C3ToU8C3<HSVToRGB<RGBToHSV<U8C3ToF32C3<BGRToRGB<Image</home/codeplay/Downloads/lena.jpg>>>>>>>>
```

Once we have created our expression tree, we can invoke `vp.run()` to execute it. What happends under the hood when `run()` is invoked is that the expression tree is flattened to a sequence of operations, which is used to generated highly optimized C++ code that is compiled and executed. Let's try it:

```python
In [13]: vp.run(output)
```

A window will open to display the result of the expression tree. Press any button to close the window and return to the python interface. The first time that an expression tree is executed, there will be a small delay while the runtime compiles it to native code. This compiled code is then cached so that future invocations of the expression tree will occur instantaneously.

And that's the end of this short tutorial. You now have a working VisionCpp environment, and know how to construct and execute your own expression trees. For further information on the kinds of operations that VisionCpp supports, see the full [API Documentation](http://chriscummins.cc/visioncpp/api.html).
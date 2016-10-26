Python interface for VisionCpp
==============================

|Build Status| |Coverage Status| |Documentation Status|

Provides a python interface for GPU-accelerated vision processing using
VisionCpp.

Requirements
------------

-  python 2.7 or >= 3.3.
-  `ComputeCpp <https://www.codeplay.com/products/computesuite/computecpp>`__
-  `clang-format <http://llvm.org/releases/download.html>`__
   *(optional)*

Installation
------------

::

    $ pip install visioncpp

To build from source:

.. code:: sh

    $ virtualenv env
    $ source env/bin/activate
    (env) $ python ./setup.py install

Get Started
-----------

.. code:: sh

    $ python

.. code:: py

    >>> import visioncpp as vp
    >>> vp.init("~/ComputeCpp-CE-0.1-Linux")  # path to your ComputeCpp package
    >>> a = vp.Image("examples/lena.jpg")
    >>> b = vp.show(a)
    >>> vp.run(b)

See ``examples/`` directory for more information.

.. |Build Status| image:: https://travis-ci.org/ChrisCummins/visioncpp.svg?branch=development
   :target: https://travis-ci.org/ChrisCummins/visioncpp
.. |Coverage Status| image:: https://coveralls.io/repos/github/ChrisCummins/visioncpp/badge.svg?branch=development
   :target: https://coveralls.io/github/ChrisCummins/visioncpp?branch=development
.. |Documentation Status| image:: https://readthedocs.org/projects/visioncpp/badge/?version=latest
   :target: http://visioncpp.readthedocs.io/en/latest/?badge=latest

from setuptools import setup
from pip.req import parse_requirements


def visioncpp_headers():
    """
    Find the VisionCpp headers using a recursive walk.

    Python didn't support glob "**/" until 3.5.

    Returns:
        str[]: Relative path to VisionCpp headers.
    """
    import os
    cwd = os.getcwd()

    # Change to the module root directory, since package_data paths must be
    # relative to this.
    module_root = "visioncpp"
    os.chdir(module_root)

    # Recursively list header files.
    header_root = "lib/include/"
    header_extension = ".hpp"
    visioncpp_headers = [
        os.path.join(dp, f) for dp, dn, filenames
        in os.walk(header_root, followlinks=True)
        for f in filenames if os.path.splitext(f)[1] == header_extension]

    # Restore the working directory.
    os.chdir(cwd)

    return visioncpp_headers


install_reqs = parse_requirements('./requirements.txt', session=False)
requirements = [str(ir.req) for ir in install_reqs]


setup(
    name='visioncpp',
    version='0.1.0',
    description='Fast, GPU-accelerated computer vision and image processing',
    url=('https://github.com/ChrisCummins/visioncpp/tree/'
         'development/contrib/python'),
    author='Chris Cummins',
    author_email='chrisc.101@gmail.com',
    license='Apache License, Version 2.0',
    packages=['visioncpp'],
    package_data={'visioncpp': visioncpp_headers()},
    scripts=[],
    test_suite='nose.collector',
    tests_require=['nose'],
    keywords=[
        'vision',
        'image processing',
        'gpu',
        'sycl',
        'computecpp',
        'machine learning'
    ],
    install_requires=requirements,
    data_files=[],
    zip_safe=False
)

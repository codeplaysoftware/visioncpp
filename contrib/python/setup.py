from setuptools import setup

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


setup(
    name='visioncpp',
    version='0.0.1',
    description='',
    download_url='https://github.com/ChrisCummins/visioncpp/tarball/0.0.1',
    url='https://github.com/codeplaysoftware/visioncpp',
    author='Chris Cummins',
    author_email='chrisc.101@gmail.com',
    license='Apache License, Version 2.0',
    packages=['visioncpp'],
    package_data={'visioncpp': visioncpp_headers()},
    scripts=[],
    test_suite='nose.collector',
    tests_require=['nose'],
    install_requires=[
        "pkgconfig == 1.1.0",
        "Send2Trash == 1.3.0",
    ],
    data_files=[],
    zip_safe=False
)

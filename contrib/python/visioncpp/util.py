"""
Utility functions for Python VisionCpp interface.
"""
import struct
import imghdr

def foreach(func, iterable):
    """
    Call function for each element of an iterable. Basically, it's a list
    comprehension without a return value.

    Arguments:
        func (function): Function to call.
        iterable (iterable): Sequence of items.
    """
    for item in iterable:
        func(item)


def call_if_attribute(obj, attr, *args, **kwargs):
    """
    Call object method, if it exists.

    Arguments:
        obj (object): The object.
        attr (str): The name of the object method.
        *args (optional): Arguments for method.
        **kwargs (optional): Keyword arguments for method.

    Returns:
        Return value of calling method, or None if object does not have method.
    """
    op = getattr(obj, attr, None)
    if callable(op):
        return op(*args, **kwargs)


def get_attribute(obj, attr):
    """
    Return object attribute value, if it exists.

    Arguments:
        obj (object): The object.
        attr (str): The name of the object attribute.
    """
    at = getattr(obj, attr, None)
    if at:
        return at
    else:
        return None


def get_image_size(fname):
    """
    Return the size of an image, in pixels.

    Arguments:
        fname (str): Path to image, either png, gif, or jpeg type.

    Returns:
        (int, int): The dimensions of the image, width x height.
    """
    import visioncpp as vp  # needed for VisionCppException type
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                raise vp.VisionCppException('failed to read image')
        else:
            raise vp.VisionCppException('unsupported image type')
        return width, height

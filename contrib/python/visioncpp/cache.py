"""
VisionCpp caching mechanism.

Provides a mechanism to cache compiled binaries to amortize compiler
overheads.
"""
import visioncpp as vp

import os

from hashlib import sha256
from shutil import move
from send2trash import send2trash

# Cache directory.
cacheroot = os.path.expanduser("~/.cache/visioncpp")


def _bin_cache_path():
    """ Get path to binaries cache. """
    return os.path.join(cacheroot, "bin")


def init(path=None):
    """
    Initialize cache.

    This function is automatically called before any caching operation which
    would require it, however users may still call this to set an alternate
    cache location.

    Arguments:
        path (str, optional): Path to new cache location.

    Returns:
        str: Path to cache directory.
    """
    global cacheroot

    if path is None:
        if not os.path.exists(_bin_cache_path()):
            os.makedirs(_bin_cache_path())
        return cacheroot
    else:
        cacheroot = os.path.expanduser(path)
        return init()


def get_uid(code):
    """
    Compute an ID for a piece of code.

    Arguments:
        code (str): The code to generate the uid for.

    Returns:
        str: UID.
    """
    return sha256(str(code).encode("utf-8")).hexdigest()


def is_cached(uid):
    """
    Returns whether there is a cache entry for the given ID.

    Arguments:
        uid (str): UID.

    Returns:
        bool: True if cached.
    """
    return os.path.exists(os.path.join(_bin_cache_path(), uid))


def load(uid):
    """
    Returns path to cached item.

    Arguments:
        uid (str): UID.

    Returns:
        str: Path to cached item.
    """
    return os.path.join(_bin_cache_path(), uid)


def emplace(uid, path):
    """
    Move a binary into cache.

    Arguments:
        uid (str): UID.
        path (str): Path to file.

    Returns:
        str: Path to cached item.
    """
    init()
    dest_path = os.path.join(_bin_cache_path(), uid)

    assert(os.path.exists(path))
    assert(not os.path.exists(dest_path))

    move(path, dest_path)
    return dest_path


def empty():
    """
    Empty the cache.

    Moves the cache directory to the platform-specific Trash.
    """
    send2trash(cacheroot)

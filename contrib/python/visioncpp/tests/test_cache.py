from __future__ import print_function

from unittest import TestCase

import visioncpp as vp
from visioncpp import cache


def _create_test_file(path, contents):
	with open(path, "w") as outfile:
		print(contents, end="", file=outfile)


class test_cache(TestCase):
    def test_init(self):
        self.assertEqual(cache.get_uid("test"), "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08")
        self.assertEqual(cache.get_uid("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")

    def test_is_cached(self):
    	self.assertEqual(cache.is_cached("abc"), False)
    	_create_test_file("abc.txt", "foobar")
    	cache.emplace("abc", "abc.txt")
    	self.assertEqual(cache.is_cached("abc"), True)
    	cache.empty()
    	self.assertEqual(cache.is_cached("abc"), False)

    def test_load(self):
    	_create_test_file("abc.txt", "foobar")
    	cache.emplace("abc", "abc.txt")
    	path = cache.load("abc")
    	with open(path) as infile:
	    	self.assertEqual(infile.read(), "foobar")
    	cache.empty()

    def test_init(self):
    	_create_test_file("abc.txt", "foobar")
    	cache.emplace("abc", "abc.txt")
    	path = cache.cacheroot
    	cache.init("different-path")
    	self.assertEqual(cache.is_cached("abc"), False)
    	cache.init(path)
    	self.assertEqual(cache.is_cached("abc"), True)
    	cache.empty()
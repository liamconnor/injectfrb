from setuptools import setup, Extension
import numpy as np

VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_POINT = 0
VERSION_DEV = 1

VERSION = "%d.%d.%d" % (VERSION_MAJOR, VERSION_MINOR, VERSION_POINT)
if VERSION_DEV:
    VERSION = VERSION + ".dev%d" % VERSION_DEV

SCRIPTS = ["injectfrb/inject_gaussian_fil.py",
	   "injectfrb/inject_frb.py"]

setup(
    name = "injectfrb",
    version = VERSION,
    packages = ["injectfrb"],
    scripts = SCRIPTS,
    install_requires = ['numpy'],
    # metadata for upload to PyPI
    author = "Liam Connor",
    author_email = "liam.dean.connor@gmail.com",
    description = "Software for simulating fast radio bursts",
    license = "GPL v2.0",
    url="http://github.com/liamconnor/injectfrb",
)

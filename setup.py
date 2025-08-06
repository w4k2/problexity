#! /usr/bin/env python
"""Toolbox for problem complexity."""
from __future__ import absolute_import

import os
# read the contents of your README file
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# get __version__ from _version.py
ver_file = os.path.join("problexity", "_version.py")
with open(ver_file) as f:
    exec(f.read())

DISTNAME = "problexity"
DESCRIPTION = "The problexity module is an open-source python library containing the implementation of measures describing the complexity of the classification problem."
MAINTAINER = "J. Komorniczak"
MAINTAINER_EMAIL = "joanna.komorniczak@vp.pl"
URL = "https://w4k2.github.io/problexity/"
LICENSE = "GPL-3.0"
DOWNLOAD_URL = "https://github.com/w4k2/problexity"
VERSION = __version__
INSTALL_REQUIRES = ["numpy", "scipy", "scikit-learn", "matplotlib", "igraph", "pandas"]
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=CLASSIFIERS,
)

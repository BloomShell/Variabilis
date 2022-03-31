from setuptools import setup
import numpy
from Cython.Build import cythonize

setup(
    ext_modules= cythonize("recursions.pyx",
    compiler_directives={"boundscheck": False, "nonecheck": False, "cdivision": True}),
    include_dirs=[numpy.get_include()]
)
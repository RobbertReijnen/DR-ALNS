from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extension = Extension(
    "repair_operators",
    sources=["repair_operators.pyx"],
    include_dirs=[numpy.get_include()],
)

setup(
    name="repair_operators",
    ext_modules=cythonize([extension])
)